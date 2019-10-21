import tensorflow as tf
from scipy.special import logit

from lib_yolo import data


def logit(x):
    """
    inverse of sigmoid function
    """
    return - tf.log((1. / x) - 1.)


def create_prior_data(det_layers):
    # create priors
    bboxes = bbox_areas = cx = cy = pw = ph = lw = lh = center_x = center_y = None
    for layer in det_layers:
        bboxes_, bbox_areas_, cx_, cy_, pw_, ph_, lw_, lh_, center_x_, center_y_ = data.create_prior_data(layer)

        bboxes_ = tf.constant(bboxes_, dtype=tf.float32)
        bbox_areas_ = tf.constant(bbox_areas_, dtype=tf.float32)
        cx_ = tf.constant(cx_, dtype=tf.float32)
        cy_ = tf.constant(cy_, dtype=tf.float32)
        pw_ = tf.constant(pw_, dtype=tf.float32)
        ph_ = tf.constant(ph_, dtype=tf.float32)
        center_x_ = tf.constant(center_x_, dtype=tf.float32)
        center_y_ = tf.constant(center_y_, dtype=tf.float32)

        bboxes_ = tf.reshape(bboxes_, shape=[-1, 4])
        bbox_areas_ = tf.reshape(bbox_areas_, shape=[-1])
        cx_ = tf.reshape(cx_, shape=[-1])
        cy_ = tf.reshape(cy_, shape=[-1])
        pw_ = tf.reshape(pw_, shape=[-1])
        ph_ = tf.reshape(ph_, shape=[-1])
        lw_ = tf.reshape(lw_, shape=[-1])
        lh_ = tf.reshape(lh_, shape=[-1])
        center_x_ = tf.reshape(center_x_, shape=[-1])
        center_y_ = tf.reshape(center_y_, shape=[-1])

        if bboxes is None:
            bboxes = bboxes_
            bbox_areas = bbox_areas_
            cx = cx_
            cy = cy_
            pw = pw_
            ph = ph_
            lw = lw_
            lh = lh_
            center_x = center_x_
            center_y = center_y_
        else:
            bboxes = tf.concat([bboxes, bboxes_], axis=0)
            bbox_areas = tf.concat([bbox_areas, bbox_areas_], axis=0)
            cx = tf.concat([cx, cx_], axis=0)
            cy = tf.concat([cy, cy_], axis=0)
            pw = tf.concat([pw, pw_], axis=0)
            ph = tf.concat([ph, ph_], axis=0)
            lw = tf.concat([lw, lw_], axis=0)
            lh = tf.concat([lh, lh_], axis=0)
            center_x = tf.concat([center_x, center_x_], axis=0)
            center_y = tf.concat([center_y, center_y_], axis=0)

    return {
        'bboxes': bboxes,
        'bbox_areas': bbox_areas,
        'cx': cx,
        'cy': cy,
        'pw': pw,
        'ph': ph,
        'lw': lw,
        'lh': lh,
        'center_x': center_x,  # TODO unused
        'center_y': center_y,  # TODO unused
    }


def encode_boxes(bboxes, labels, det_layers, ign_thresh):
    prior_data = create_prior_data(det_layers)

    # initialize output
    total_box_cnt = 0
    layer_sizes = []
    for layer in det_layers:
        boxes_per_cell = len(layer.priors)
        layer_sizes.append(layer.h * layer.w * boxes_per_cell)
        total_box_cnt += layer_sizes[-1]

    # loc = tf.zeros(shape=(total_box_cnt, 4), dtype=tf.float32)
    loc_x = tf.zeros(shape=(total_box_cnt,), dtype=tf.float32)
    loc_y = tf.zeros(shape=(total_box_cnt,), dtype=tf.float32)
    loc_w = tf.zeros(shape=(total_box_cnt,), dtype=tf.float32)
    loc_h = tf.zeros(shape=(total_box_cnt,), dtype=tf.float32)
    obj = tf.zeros(shape=(total_box_cnt,), dtype=tf.float32)
    cls = tf.zeros(shape=(total_box_cnt,), dtype=tf.int32)
    ign = tf.ones(shape=(total_box_cnt,), dtype=tf.float32)

    ones_float = tf.ones(shape=(total_box_cnt,), dtype=tf.float32)
    ones_int = tf.ones(shape=(total_box_cnt,), dtype=tf.int32)
    zeros = tf.zeros(shape=(total_box_cnt,), dtype=tf.float32)

    w = bboxes[..., 3] - bboxes[..., 1]
    h = bboxes[..., 2] - bboxes[..., 0]
    x = (bboxes[..., 3] + bboxes[..., 1]) / 2.
    y = (bboxes[..., 2] + bboxes[..., 0]) / 2.

    def loop_condition(idx_, *vargs):
        return tf.less(idx_, tf.shape(labels)[0])

    # calc overlaps between gt bboxes and priors
    def loop_body(idx_, loc_x_, loc_y_, loc_w_, loc_h_, obj_, cls_, ign_):
        bbox = bboxes[idx_]

        # TODO can this be optimized? x[idx] * ones
        dist_to_cell_center_x = prior_data['lw'] * (x[idx_] - prior_data['cx'])
        dist_to_cell_center_y = prior_data['lh'] * (y[idx_] - prior_data['cy'])
        x_obj_mask = tf.logical_and(tf.greater_equal(dist_to_cell_center_x, 0), tf.less_equal(dist_to_cell_center_x, 1))
        y_obj_mask = tf.logical_and(tf.greater_equal(dist_to_cell_center_y, 0), tf.less_equal(dist_to_cell_center_y, 1))
        obj_mask = tf.logical_and(x_obj_mask, y_obj_mask)

        # calc best iou score
        iou = calc_iou(bbox, prior_data)
        best_ious = tf.greater_equal(iou, tf.reduce_max(iou))  # TODO performance

        # use location and iou to determine the correct prior and cell
        obj_mask = tf.logical_and(best_ious, obj_mask)

        # check if we got at least one obj
        # assertion = tf.assert_greater(tf.reduce_sum(tf.where(obj_mask, ones_float, zeros)),
        #                               0.5, message='ERROR, skipped obj')

        ign_mask = tf.greater_equal(iou, ign_thresh)

        # with tf.control_dependencies([assertion]):
        eps = 1e-7
        loc_x_ = tf.where(obj_mask, logit(tf.clip_by_value(dist_to_cell_center_x, eps, 1 - eps)), loc_x_)
        loc_y_ = tf.where(obj_mask, logit(tf.clip_by_value(dist_to_cell_center_y, eps, 1 - eps)), loc_y_)

        loc_w_ = tf.where(obj_mask, tf.log(tf.maximum(w[idx_] / prior_data['pw'], eps)), loc_w_)
        loc_h_ = tf.where(obj_mask, tf.log(tf.maximum(h[idx_] / prior_data['ph'], eps)), loc_h_)
        cls_ = tf.where(obj_mask, labels[idx_] * ones_int, cls_)
        obj_ = tf.where(obj_mask, ones_float, obj_)  # TODO make conditional on cls == ignore
        ign_ = tf.where(ign_mask, zeros, ign_)

        idx_ += 1

        return idx_, loc_x_, loc_y_, loc_w_, loc_h_, obj_, cls_, ign_

    idx = 0
    [idx, loc_x, loc_y, loc_w, loc_h, obj, cls, ign] = tf.while_loop(loop_condition, loop_body,
                                                                     [idx, loc_x, loc_y, loc_w, loc_h, obj, cls,
                                                                      ign])

    loc = tf.stack([loc_x, loc_y, loc_w, loc_h], axis=1)  # maybe this is unnecessary
    ign = tf.maximum(ign, obj)

    loc = tf.split(loc, layer_sizes, axis=0)
    cls = tf.split(cls, layer_sizes, axis=0)
    obj = tf.split(obj, layer_sizes, axis=0)
    ign = tf.split(ign, layer_sizes, axis=0)

    encoded = []
    for i, layer in enumerate(det_layers):
        shape = [layer.h, layer.w, len(layer.priors)]
        encoded.append({
            'loc': tf.reshape(loc[i], shape=[*shape, 4]),
            'cls': tf.reshape(cls[i], shape=shape),
            'obj': tf.reshape(obj[i], shape=shape),
            'ign': tf.reshape(ign[i], shape=shape),
        })

    return encoded


def calc_iou(ref_bbox, prior_data):
    bboxes = prior_data['bboxes']
    bboxes_area = prior_data['bbox_areas']

    int_ymin = tf.maximum(bboxes[..., 0], ref_bbox[0])
    int_xmin = tf.maximum(bboxes[..., 1], ref_bbox[1])
    int_ymax = tf.minimum(bboxes[..., 2], ref_bbox[2])
    int_xmax = tf.minimum(bboxes[..., 3], ref_bbox[3])
    h = tf.maximum(int_ymax - int_ymin, 0.)
    w = tf.maximum(int_xmax - int_xmin, 0.)

    inter = h * w
    union = bboxes_area - inter + ((ref_bbox[2] - ref_bbox[0]) * (ref_bbox[3] - ref_bbox[1]))
    iou = tf.div(inter, union)
    return iou
