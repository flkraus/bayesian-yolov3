# This file contains mostly numpy reference implementations for ground truth bbox encoding for the yolo loss.

import numpy as np
from scipy.special import logit, expit


class Box:
    def __init__(self):
        self.xmin = None
        self.ymin = None
        self.xmax = None
        self.ymax = None

        self.x_center = None
        self.y_center = None
        self.w = None
        self.h = None

        self.area = None
        self.cls = None

    def __repr__(self):
        return '<Box - x: {}, y: {}, w: {}, h: {}, label: {}>'.format(self.x_center, self.y_center, self.h, self.w,
                                                                      self.cls)

    @classmethod
    def from_corners(cls, xmin, ymin, xmax, ymax, label=-1):
        box = cls()

        box.xmin = xmin
        box.ymin = ymin
        box.xmax = xmax
        box.ymax = ymax

        box.x_center = (box.xmin + box.xmax) / 2.
        box.y_center = (box.ymin + box.ymax) / 2.
        box.w = box.xmax - box.xmin
        box.h = box.ymax - box.ymin

        box.area = box.w * box.h

        box.cls = label
        return box

    @classmethod
    def from_tf_image_format(cls, ymin, xmin, ymax, xmax, label=-1):
        return cls.from_corners(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, label=label)

    @classmethod
    def from_width_and_height(cls, x_center, y_center, w, h, label=-1):
        box = cls()

        box.x_center = x_center
        box.y_center = y_center
        box.w = w
        box.h = h

        w2 = w / 2.
        h2 = h / 2.
        box.xmin = box.x_center - w2
        box.ymin = box.y_center - h2
        box.xmax = box.x_center + w2
        box.ymax = box.y_center + h2

        box.area = box.w * box.h

        box.cls = label
        return box


class Cell:
    def __init__(self, row, col):
        self.row = row
        self.col = col

    def __repr__(self):
        return '<Cell - row: {}, col: {}>'.format(self.row, self.col)


class Prior:
    def __init__(self, h, w):
        self.h = h
        self.w = w

    def __repr__(self):
        return '<Prior - h: {}, w: {}>'.format(self.h, self.w)


class DetLayerInfo:
    def __init__(self, h, w, priors):
        self.h = h
        self.w = w
        self.priors = priors

    def __repr__(self):
        return '<DetLayerInfo - h: {}, w: {}, priors: {}>'.format(self.h, self.w, self.priors)


def create_prior_box_grid(det_layer):
    boxes_per_cell = len(det_layer.priors)
    prior_box_grid = np.zeros((det_layer.h, det_layer.w, boxes_per_cell, 4))
    for row in range(det_layer.h):
        for col in range(det_layer.w):
            for box, prior in enumerate(det_layer.priors):
                y_center = (row + 0.5) / det_layer.h
                x_center = (col + 0.5) / det_layer.w
                h2 = prior.h / 2.
                w2 = prior.w / 2.

                ymin = y_center - h2
                xmin = x_center - w2
                ymax = y_center + h2
                xmax = x_center + w2

                prior_box_grid[row, col, box, :] = [ymin, xmin, ymax, xmax]  # tf.image bbox format
    return prior_box_grid


def create_prior_data(det_layer):
    boxes_per_cell = len(det_layer.priors)

    bboxes = np.zeros((det_layer.h, det_layer.w, boxes_per_cell, 4), dtype=np.float32)
    bbox_areas = np.zeros((det_layer.h, det_layer.w, boxes_per_cell), dtype=np.float32)
    cx = np.zeros((det_layer.h, det_layer.w, boxes_per_cell), dtype=np.float32)
    cy = np.zeros((det_layer.h, det_layer.w, boxes_per_cell), dtype=np.float32)
    pw = np.zeros((det_layer.h, det_layer.w, boxes_per_cell), dtype=np.float32)
    ph = np.zeros((det_layer.h, det_layer.w, boxes_per_cell), dtype=np.float32)
    lw = np.ones((det_layer.h, det_layer.w, boxes_per_cell), dtype=np.float32) * det_layer.w
    lh = np.ones((det_layer.h, det_layer.w, boxes_per_cell), dtype=np.float32) * det_layer.h
    center_x = np.zeros((det_layer.h, det_layer.w, boxes_per_cell), dtype=np.float32)
    center_y = np.zeros((det_layer.h, det_layer.w, boxes_per_cell), dtype=np.float32)

    prior_areas = [p.h * p.w for p in det_layer.priors]

    for row in range(det_layer.h):
        for col in range(det_layer.w):
            for box, prior in enumerate(det_layer.priors):
                assert 0 <= prior.w <= 1, 'prior width must be specified as a number between 0 and 1'
                assert 0 <= prior.h <= 1, 'prior height must be specified as a number between 0 and 1'
                y_center = (row + 0.5) / det_layer.h
                x_center = (col + 0.5) / det_layer.w
                h2 = prior.h / 2.
                w2 = prior.w / 2.

                ymin = y_center - h2
                xmin = x_center - w2
                ymax = y_center + h2
                xmax = x_center + w2

                cx[row, col, box] = col / float(det_layer.w)
                cy[row, col, box] = row / float(det_layer.h)
                pw[row, col, box] = prior.w
                ph[row, col, box] = prior.h
                center_x[row, col, box] = x_center
                center_y[row, col, box] = y_center

                bboxes[row, col, box, :] = [ymin, xmin, ymax, xmax]  # tf.image bbox format
                bbox_areas[row, col, box] = prior_areas[box]
    return bboxes, bbox_areas, cx, cy, pw, ph, lw, lh, center_x, center_y  # TODO this is ugly


def calc_gt(gt_boxes, det_layers):
    gt = []
    for layer in det_layers:
        boxes_per_cell = len(layer.priors)
        gt.append({
            'loc': np.zeros((layer.h, layer.w, boxes_per_cell, 4)),
            'obj': np.zeros((layer.h, layer.w, boxes_per_cell)),
            'cls': np.zeros((layer.h, layer.w, boxes_per_cell)),
            'fp': np.zeros((layer.h, layer.w, boxes_per_cell)),
            'ignore': np.ones((layer.h, layer.w, boxes_per_cell)),
        })

    prior_grids = []
    for layer in det_layers:
        prior_grids.append(create_prior_box_grid(layer))

    used_cells = {}

    for gt_box in gt_boxes:
        res = find_responsible_layer_and_prior(det_layers, gt_box)
        l_idx = res['layer']
        p_idx = res['prior']
        layer = det_layers[l_idx]
        prior = layer.priors[p_idx]
        cell = find_responsible_cell(layer, gt_box)

        used_cells[(l_idx, p_idx, cell.row, cell.col)] = used_cells.get((l_idx, p_idx, cell.row, cell.col), 0) + 1

        cx = cell.col / float(layer.w)
        cy = cell.row / float(layer.h)
        tx = logit(gt_box.x_center - cx)
        ty = logit(gt_box.y_center - cy)
        if tx < -100 or tx > 100:
            assert False
        if ty < -100 or ty > 100:
            assert False
        tw = np.log(gt_box.w / prior.w)
        th = np.log(gt_box.h / prior.h)

        gt[l_idx]['loc'][cell.row, cell.col, p_idx, :] = [tx, ty, tw, th]
        gt[l_idx]['obj'][cell.row, cell.col, p_idx] = 1
        gt[l_idx]['cls'][cell.row, cell.col, p_idx] = gt_box.cls
        gt[l_idx]['fp'][cell.row, cell.col, p_idx] = 1

        # calc iou for all prior boxes for all layers with the gt_box
        ious = iou_multiboxes(gt_box, prior_grids)
        for i in range(len(det_layers)):
            gt[i]['ignore'][ious[i] > 0.7] = 0  # TODO ignore threshold

    for i in range(len(det_layers)):
        gt[i]['ignore'] = np.maximum(gt[i]['ignore'], gt[i]['fp'])

    return gt, used_cells


def iou_multiboxes(gt_box, prior_grids):
    ious = []
    for pg in prior_grids:
        iou_grid = np.zeros(pg.shape[:3])
        for row in range(pg.shape[0]):
            for col in range(pg.shape[1]):
                for box in range(pg.shape[2]):
                    iou_grid[row, col, box] = iou(gt_box, Box.from_tf_image_format(*pg[row, col, box, :]))
        ious.append(iou_grid)

    return ious


def find_responsible_cell(det_layer, gt_box):
    row = int(det_layer.h * gt_box.y_center)
    col = int(det_layer.w * gt_box.x_center)
    return Cell(row, col)


def find_responsible_layer_and_prior(det_layers, gt_box):
    gt_box = Box.from_width_and_height(0, 0, w=gt_box.w, h=gt_box.h)
    best_iou = 0
    best_layer = None
    for l_idx, layer in enumerate(det_layers):
        ious = [iou(gt_box, Box.from_width_and_height(0, 0, w=prior.w, h=prior.h)) for prior in layer.priors]
        if np.max(ious) > best_iou:
            best_prior = np.argmax(ious)
            best_layer = l_idx
            best_iou = np.max(ious)

    assert best_layer is not None
    assert best_iou > 0
    return {'layer': best_layer, 'prior': best_prior}


def iou(b1, b2):
    intersection = intersect(b1, b2)
    if intersect == 0:  # TODO use np.is_close?
        return 0.0

    union = b1.area + b2.area - intersection

    return intersection / union


def intersect(b1, b2):
    """
    :param b1: Box
    :param b2: Box
    :return:
    """

    xmin = np.maximum(b1.xmin, b2.xmin)
    ymin = np.maximum(b1.ymin, b2.ymin)
    xmax = np.minimum(b1.xmax, b2.xmax)
    ymax = np.minimum(b1.ymax, b2.ymax)

    if xmax <= xmin:
        return 0.0

    if ymax <= ymin:
        return 0.0

    intersection_box = Box.from_corners(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
    return intersection_box.area


def loc_to_boxes(loc, cls, fp, priors):
    lh, lw, boxes_per_cell = fp.shape
    boxes = []
    for row in range(lh):
        for col in range(lw):
            for box in range(boxes_per_cell):
                if fp[row, col, box] == 1:
                    cx = col / float(lw)
                    cy = row / float(lh)
                    x_center = expit(loc[row, col, box, 0]) + cx
                    y_center = expit(loc[row, col, box, 1]) + cy
                    w = np.exp(loc[row, col, box, 2]) * priors[box].w
                    h = np.exp(loc[row, col, box, 3]) * priors[box].h

                    label = cls[row, col, box]

                    boxes.append(Box.from_width_and_height(x_center=x_center, y_center=y_center, w=w, h=h, label=label))
    return boxes


def loc_to_tf_records_format(loc, cls, fp, priors):
    ymin, xmin, ymax, xmax, labels = [], [], [], [], []
    boxes = loc_to_boxes(loc, cls, fp, priors)
    for box in boxes:
        ymin.append(box.ymin)
        xmin.append(box.xmin)
        ymax.append(box.ymax)
        xmax.append(box.xmax)
        labels.append(box.cls)
    return [ymin, xmin, ymax, xmax], labels


def create_boxes_from_tf_records_format(boxes):
    out = []
    for i in range(len(boxes['ymin'])):
        out.append(Box.from_corners(xmin=boxes['xmin'][i],
                                    ymin=boxes['ymin'][i],
                                    xmax=boxes['xmax'][i],
                                    ymax=boxes['ymax'][i],
                                    label=boxes['cls'][i], ))

    return out
