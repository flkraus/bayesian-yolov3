import glob
import logging
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import inference_aleatoric
import inference_epistemic
import inference_standard_yolov3
from lib_yolo import yolov3


def box_op_standard(model):
    bbox = inference_standard_yolov3.concat_bbox([det_layer.bbox for det_layer in model.det_layers])
    nms = inference_standard_yolov3.nms(bbox, model)
    nms = nms[0, ...]
    return nms


def box_op_aleatoric(model):
    bbox = inference_aleatoric.concat_bbox([det_layer.bbox for det_layer in model.det_layers])
    nms = inference_aleatoric.nms(bbox, model)
    nms = nms[0, ...]
    return nms


def box_op_bayes(model):
    bbox = inference_epistemic.concat_bbox([det_layer.bbox for det_layer in model.det_layers])
    nms = inference_epistemic.nms(bbox, model)
    return nms


def filter_boxes(boxes, obj_idx, thresh):
    return [box for box in boxes if box[obj_idx] > thresh]


def preproces_boxes(img_size, boxes, obj_idx, cls_start_idx, cls_cnt, config, cls_mapping=None):
    out = []
    for box in boxes:
        cls_idx = np.argmax(box[cls_start_idx:cls_start_idx + cls_cnt])
        if config['implicit_background_class']:
            cls_idx += 1
        if cls_mapping:
            cls = cls_mapping[cls_idx]
        else:
            cls = cls_idx

        cls_score = box[cls_idx + cls_start_idx]
        out.append({
            'cls': cls,
            'score': box[obj_idx] * cls_score,
            'obj_score': box[obj_idx],
            'cls_score': cls_score,
            'y0': np.clip(box[0], 0, 1) * img_size[0],
            'x0': np.clip(box[1], 0, 1) * img_size[1],
            'y1': np.clip(box[2], 0, 1) * img_size[0],
            'x1': np.clip(box[3], 0, 1) * img_size[1],
        })

    return out


def draw_boxes(img, boxes, color=(43, 219, 216), thickness=1):
    color = np.array(color) / 255.
    for box in boxes:
        text = '{} {:4.3f}'.format(box['cls'], box['score'])
        size = 0.5
        cv2.putText(img, text, (int(box['x0']), int(box['y0'])), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness)

        cv2.rectangle(img, (int(box['x0']), int(box['y0'])), (int(box['x1']), int(box['y1'])), color, thickness)


def load_img(config, img_size, filename):
    img = plt.imread(filename)  # loads image as np.float32 array

    if config['crop']:
        y = (img.shape[0] - img_size[0]) // 2
        x = (img.shape[1] - img_size[1]) // 2
        img = img[y:y + img_size[0], x:x + img_size[1], :]

    img = np.expand_dims(img, axis=0)
    return img


def load_model(sess, config, model_cls):
    if model_cls == yolov3.bayesian_yolov3_aleatoric:
        config['inference_mode'] = True

    yolo = model_cls(config)
    img_tensor = tf.placeholder(tf.float32, shape=(1, *yolo.img_size))
    model = yolo.init_model(inputs=img_tensor, training=False).get_model()

    checkpoints = os.path.join(config['checkpoint_path'], config['run_id'])
    if config['step'] == 'last':
        checkpoint = tf.train.latest_checkpoint(checkpoints)
    else:
        checkpoint = None
        for cp in os.listdir(checkpoints):
            if cp.endswith('-{}.meta'.format(config['step'])):
                checkpoint = os.path.join(checkpoints, os.path.splitext(cp)[0])
                break
        assert checkpoint is not None, 'could not find checkpoint'

    tf.train.Saver().restore(sess, checkpoint)

    return model, img_tensor


def do_it(files, thresh, config, model_cls, cls_mapping):
    box_op = {
        yolov3.yolov3: box_op_standard,
        yolov3.yolov3_aleatoric: box_op_aleatoric,
        yolov3.bayesian_yolov3_aleatoric: box_op_bayes,
    }[model_cls]

    with tf.Session(config=tf.ConfigProto(device_count={'GPU': 1})) as sess:
        model, img_tensor = load_model(sess, config, model_cls)
        img_size = img_tensor.shape.as_list()[1:]
        for file in files:
            img = load_img(config, img_size, file)
            boxes, = sess.run([box_op(model)], feed_dict={img_tensor: img})
            boxes = filter_boxes(boxes, model.obj_idx, thresh)
            boxes = preproces_boxes(img_size, boxes, model.obj_idx, model.cls_start_idx, model.cls_cnt,
                                    config, cls_mapping=cls_mapping)

            img = img[0, ...]
            draw_boxes(img, boxes)
            logging.info('{}: {}'.format(os.path.basename(file), boxes))

            plt.imshow(img)
            plt.show()
            # plt.imsave(filename, img)  #


def main():
    config = {
        'checkpoint_path': './checkpoints/',
        'run_id': 'epi_ale',  # edit
        'step': 'last',  # edit: int or 'last'
        'crop_img_size': [768, 1440, 3],
        'full_img_size': [1024, 1920, 3],  # edit if not ecp
        'cls_cnt': 2,  # edit if not ecp
        'T': 35,  # edit if OOM error, only relevant for bayesian model
        'cpu_thread_cnt': 10,
        'freeze_darknet53': False,  # actual value irrelevant
        'crop': False,  # edit: less memory consumption if True
        'training': False,
        'aleatoric_loss': True,  # actual value irrelevant
        'priors': yolov3.ECP_9_PRIORS,  # actual value irrelevant
        'out_path': './uncertainty_visualization',  # edit
        'implicit_background_class': True,  # whether the label ids start at 1 or 0. True = 1, False = 0
    }

    class_name_mapping_implicit_background_cls = {  # edit: change if you have more or different classes, or set to None
        1: 'ped',  #
        2: 'rider',
    }

    # class_name_mapping_no_implicit_background_cls = {  # if your labels start at 0 instead of 1
    #     0: 'ped',  #
    #     1: 'rider',
    # }

    thresh = 0.1  # edit

    files = glob.glob('./test_images/*')  # edit

    # EDIT: chose appropriate model class
    # model_cls = yolov3.yolov3
    # model_cls = yolov3.yolov3_aleatoric
    model_cls = yolov3.bayesian_yolov3_aleatoric

    do_it(files, thresh, config, model_cls, class_name_mapping_implicit_background_cls)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s, pid: %(process)d, %(levelname)-8s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        )
    main()
