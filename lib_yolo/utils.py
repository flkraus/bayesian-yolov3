import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from lib_yolo import dataset_utils, yolov3, data_augmentation


def logistic(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


def _vis(sess, model, inputs, thresh, epistemic=False, feed_dict=None):
    *bboxes, img = sess.run([*[det_layer.bbox for det_layer in model.det_layers], inputs], feed_dict=feed_dict)

    boxes = []
    for bbox_all in bboxes:
        for bbox_by_prior in bbox_all:
            if epistemic:
                b = bbox_by_prior
            else:
                b = bbox_by_prior[0, ...]
            b = np.reshape(b, newshape=[b.shape[0] * b.shape[1], b.shape[2]])
            boxes.append(b)
    boxes = np.concatenate(boxes, axis=0)

    img = img[0, ...]
    img = np.expand_dims(img, 0)

    def draw_boxes(img_, boxes_):
        nms_ind = tf.image.non_max_suppression(boxes_[:, :4], boxes_[:, model.obj_idx], 1000)
        boxes_ = tf.gather(boxes_, nms_ind, axis=0)
        thresh_ind = tf.where(boxes_[:, model.obj_idx] > thresh)
        boxes_ = tf.gather(boxes_, thresh_ind, axis=0)
        boxes_ = tf.squeeze(boxes_, axis=[-2])
        boxes_ = boxes_[:, :4]
        boxes_ = tf.expand_dims(boxes_, axis=0)
        return tf.image.draw_bounding_boxes(img_, boxes_)

    draw_op = tf.cond(tf.reduce_any(boxes[:, model.obj_idx] > thresh),
                      true_fn=lambda: draw_boxes(img, boxes),
                      false_fn=lambda: img)

    result = sess.run(draw_op)

    # mng = plt.get_current_fig_manager()
    # mng.resize(*mng.window.maxsize())

    plt.imshow((255 * np.squeeze(result)).astype(np.uint8))
    plt.show()


def vis_standard(sess, model, inputs, thresh, feed_dict=None):
    _vis(sess, model, inputs, thresh, feed_dict=feed_dict)


def vis_aleatoric(sess, model, inputs, thresh, feed_dict=None):
    _vis(sess, model, inputs, thresh, feed_dict=feed_dict)


def vis_bayes(sess, model, inputs, thresh, feed_dict=None):
    _vis(sess, model, inputs, thresh, epistemic=True, feed_dict=feed_dict)


def predictions_to_boxes_numpy_reference_implementation(predictions, cls_cnt, priors, box_format='xywh'):
    # predictions = np.array(predictions, dtype=np.float32)

    batches, lh, lw, c = predictions.shape
    det_size = 2 * (4 + 1 + cls_cnt)

    result = np.zeros([batches, len(priors) * lh * lw, det_size], dtype=np.float32)

    for b in range(batches):
        result_idx = 0
        for row in range(lh):
            for col in range(lw):
                anchor_offset = 0
                for p in priors:
                    [x, y, w, h, x_var, y_var, w_var, h_var,
                     obj, log_obj_stddev, *cls_] = predictions[b, row, col, anchor_offset:anchor_offset + det_size]
                    cls = cls_[:cls_cnt]
                    log_cls_stddev = cls_[cls_cnt:]
                    cls = np.array(cls, dtype=np.float32)

                    x = (col + logistic(x)) / lw
                    y = (row + logistic(y)) / lh
                    w = (np.exp(w) * p.w)
                    h = (np.exp(h) * p.h)
                    x_var = np.exp(x_var)
                    y_var = np.exp(y_var)
                    w_var = np.exp(w_var)
                    h_var = np.exp(h_var)

                    obj = logistic(obj)
                    obj_stddev = np.exp(log_obj_stddev)
                    cls = softmax(cls)
                    cls_stddev = np.exp(log_cls_stddev)

                    if box_format == 'xywh':
                        result[b, result_idx, :] = [x, y, w, h, x_var, y_var, w_var, h_var, obj, obj_stddev,
                                                    *cls, *cls_stddev]
                    else:
                        w2 = w / 2
                        h2 = h / 2
                        x0 = x - w2
                        y0 = y - h2
                        x1 = x + w2
                        y1 = y + h2

                        result[b, result_idx, :] = [y0, x0, y1, x1, x_var, y_var, w_var, h_var, obj, obj_stddev,
                                                    *cls, *cls_stddev]

                    result_idx += 1
                    anchor_offset += det_size

    return result


def qualitative_eval(model_cls, config):
    if config['crop']:
        cropper = data_augmentation.ImageCropper(config)
        config['val']['crop_fn'] = cropper.center_crop

    if model_cls == yolov3.bayesian_yolov3_aleatoric:
        config['inference_mode'] = True
        config.setdefault('T', 20)

    model_factory = model_cls(config)

    dataset = dataset_utils.ValDataset(config, dataset_key='val')
    img, bbox, label = dataset.iterator.get_next()

    model = model_factory.init_model(inputs=img, training=False).get_model()
    with tf.Session(config=tf.ConfigProto(device_count={'GPU': 1})) as sess:
        checkpoint = config.get('resume_checkpoint', 'last')
        if checkpoint == 'last':
            checkpoint = tf.train.latest_checkpoint(os.path.join(config['checkpoint_path'], config['run_id']))
        tf.train.Saver().restore(sess, checkpoint)

        vis_fn = {
            yolov3.yolov3: _vis,
            yolov3.yolov3_aleatoric: vis_aleatoric,
            yolov3.bayesian_yolov3_aleatoric: vis_bayes,
        }[model_cls]
        for i in range(1000):
            vis_fn(sess, model, img, config['thresh'])


def add_file_logging(config, override_existing=False):
    path = os.path.join(config['log_path'], '{}.log'.format(config['run_id']))

    try:
        os.makedirs(config['log_path'])
    except IOError:
        pass

    if os.path.exists(path) and not override_existing:
        raise RuntimeError('Logging file {} already exists'.format(path))

    file_handler = logging.FileHandler(path, 'w')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s, %(levelname)-8s %(message)s',
                                  datefmt='%a, %d %b %Y %H:%M:%S')
    file_handler.setFormatter(formatter)
    logging.getLogger('').addHandler(file_handler)
