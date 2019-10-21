"""
Inference script for the yolov3.bayesian_yolov3_aleatoric class.

Produces detection files for each input image conforming to the ECP .json format.
The output of this script can be directly used by the ECP evaluation code.
"""

import json
import logging
import os
import threading
import time

import numpy as np
import tensorflow as tf

from lib_yolo import dataset_utils, yolov3


class Inference:
    def __init__(self, yolo, config):
        self.batch_size = config['batch_size']

        dataset = dataset_utils.TestingDataset(config)
        self.img_tensor, self.filename_tensor = dataset.iterator.get_next()

        checkpoints = os.path.join(config['checkpoint_path'], config['run_id'])
        if config['step'] == 'last':
            self.checkpoint = tf.train.latest_checkpoint(checkpoints)
        else:
            self.checkpoint = None
            for cp in os.listdir(checkpoints):
                if cp.endswith('-{}.meta'.format(config['step'])):
                    self.checkpoint = os.path.join(checkpoints, os.path.splitext(cp)[0])
                    break
            assert self.checkpoint is not None

        step = self.checkpoint.split('-')[-1]

        self.img_size = config['full_img_size']
        assert not config['crop']
        self.out_path = '{}_{}'.format(config['out_path'], step)
        os.makedirs(self.out_path)

        self.config = config
        self.worker_thread = None

        assert config['inference_mode']
        self.model = yolo.init_model(inputs=self.img_tensor, training=False).get_model()

        bbox = concat_bbox([self.model.det_layers[0].bbox,
                            self.model.det_layers[1].bbox,
                            self.model.det_layers[2].bbox])
        self.nms = nms(bbox, self.model)

    def run(self):
        with tf.Session(config=tf.ConfigProto(device_count={'GPU': 1})) as sess:
            tf.train.Saver().restore(sess, self.checkpoint)

            step = 0
            while True:
                try:
                    step += 1
                    self.epistemic_forward_pass(sess)

                    if step % 15 == 0:
                        logging.info('Processed {} images.'.format(step))

                except tf.errors.OutOfRangeError:
                    break
            logging.info('Processed {} images.'.format(step))

        self.worker_thread.join()

    def epistemic_forward_pass(self, sess):
        boxes, files = sess.run([self.nms, self.filename_tensor])

        if self.worker_thread:
            self.worker_thread.join()

        img_name = files[0][0].decode('utf-8')
        self.worker_thread = threading.Thread(target=self.write_ecp_json, args=(boxes, img_name))
        self.worker_thread.start()

    def write_ecp_json(self, boxes, img_name):
        out_name = '{}.json'.format(os.path.splitext(os.path.basename(img_name))[0])
        out_file = os.path.join(self.out_path, out_name)

        with open(out_file, 'w') as f:
            json.dump({
                'children': [bbox_to_ecp_format(bbox, self.img_size, self.model, self.config) for bbox in boxes],
            }, f, default=lambda x: x.tolist())


# -----------------------------------------------------------------#
#                             helpers                              #
# -----------------------------------------------------------------#

def nms(boxes, model):
    # nms ignoring classes
    nms_indices = tf.image.non_max_suppression(boxes[:, :4], boxes[:, model.obj_idx], 1000)
    all_boxes = tf.gather(boxes, nms_indices, axis=0)

    # # nms for each class individually, works only for data with 2 classes (e.g. ECP dataset)
    # # this was used to produce the results for the paper
    # nms_boxes = None
    # for cls in ['ped', 'rider']:
    #     if cls == 'ped':
    #         tmp = tf.greater(b[:, model.cls_start_idx], b[:, model.cls_start_idx + 1])
    #     elif cls == 'rider':
    #         tmp = tf.greater(b[:, model.cls_start_idx + 1], b[:, model.cls_start_idx])
    #     else:
    #         raise ValueError('invalid class: {}'.format(cls))
    #
    #     cls_indices = tf.cast(tf.reshape(tf.where(tmp), [-1]), tf.int32)
    #
    #     cls_boxes = tf.gather(b, cls_indices)
    #     ind = tf.image.non_max_suppression(cls_boxes[:, :4], cls_boxes[:, model.obj_idx], 1000)
    #     cls_boxes = tf.gather(cls_boxes, ind, axis=0)
    #
    #     if nms_boxes is None:
    #         nms_boxes = cls_boxes
    #     else:
    #         nms_boxes = tf.concat([nms_boxes, cls_boxes], axis=0)
    #
    # return nms_boxes

    return all_boxes


def bbox_to_ecp_format(bbox, img_size, model, config):
    img_height, img_width = img_size[:2]
    label_to_cls_name = {  # edit if not ECP dataset
        1: 'pedestrian',  # starts at 0 if no implicit background class
        2: 'rider',
    }
    cls_scores = bbox[model.cls_start_idx:model.cls_start_idx + model.cls_cnt]
    cls = np.argmax(cls_scores)

    cls_idx = cls
    if config['implicit_background_class']:
        cls += 1

    return {
        'y0': float(bbox[0] * img_height),
        'x0': float(bbox[1] * img_width),
        'y1': float(bbox[2] * img_height),
        'x1': float(bbox[3] * img_width),
        'x_var_epi': float(bbox[4]),
        'y_var_epi': float(bbox[5]),
        'w_var_epi': float(bbox[6]),
        'h_var_epi': float(bbox[7]),
        'x_var_ale': float(bbox[8]),  # random value for models trained without aleatoric loss.
        'y_var_ale': float(bbox[9]),  # random value for models trained without aleatoric loss.
        'w_var_ale': float(bbox[10]),  # random value for models trained without aleatoric loss.
        'h_var_ale': float(bbox[11]),  # random value for models trained without aleatoric loss.
        'total_var_epi': float(bbox[12]),  # not useful
        'total_var_ale': float(bbox[13]),  # not useful and random value for models trained without aleatoric loss.
        'score': float(bbox[model.obj_idx]) * float(bbox[model.cls_start_idx + cls_idx]),
        'obj_mutual_info': float(bbox[model.obj_idx + 1]),
        'obj_entropy': float(bbox[model.obj_idx + 2]),
        'cls_scores': cls_scores,
        'ped_score': float(bbox[17]),
        'rider_score': float(bbox[18]),
        'cls_mutual_info': float(bbox[model.cls_start_idx + model.cls_cnt]),
        'cls_entropy': float(bbox[model.cls_start_idx + model.cls_cnt + 1]),
        'layer_id': float(bbox[model.cls_start_idx + model.cls_cnt + 2]),
        'prior_id': float(bbox[model.cls_start_idx + model.cls_cnt + 3]),
        'identity': label_to_cls_name.get(cls, cls),
    }


def concat_bbox(net_out):
    bbox = None
    for det_layer in net_out:
        for prior in det_layer:
            lw, lh, det_size = prior.shape.as_list()
            tmp = tf.reshape(prior, shape=[lw * lh, det_size])
            if bbox is None:
                bbox = tmp
            else:
                bbox = tf.concat([bbox, tmp], axis=0)

    return bbox


# -----------------------------------------------------------------#
#                               main                               #
# -----------------------------------------------------------------#


def inference(config):
    assert config['batch_size'] == 1
    assert not config['crop']

    logging.info(json.dumps(config, indent=4, default=lambda x: str(x)))

    assert not config['crop']
    logging.info('----- START -----')
    start = time.time()

    yolo = yolov3.bayesian_yolov3_aleatoric(config)
    Inference(yolo, config).run()

    end = time.time()
    elapsed = int(end - start)
    logging.info('----- FINISHED in {:02d}:{:02d}:{:02d} -----'.format(elapsed // 3600,
                                                                       (elapsed // 60) % 60,
                                                                       elapsed % 60))


def main():
    config = {
        'checkpoint_path': './checkpoints',  # edit
        'run_id': 'epi_ale',  # edit
        # 'step': 500000,  # edit
        'step': 'last',  # edit
        'full_img_size': [1024, 1920, 3],  # edit if not ECP dataset
        'cls_cnt': 2,  # edit if not ECP dataset
        'batch_size': 1,
        'T': 50,  # edit if OOM errors
        'inference_mode': True,
        'cpu_thread_cnt': 24,  # edit
        'crop': False,
        'training': False,
        'aleatoric_loss': False,
        'priors': yolov3.ECP_9_PRIORS,  # edit
        'implicit_background_class': True,
        'data': {
            'path': '$HOME/data/ecp/tfrecords',  # edit
            'file_pattern': 'ecp-day-val-*-of-*',  # edit
        }
    }

    config['data']['file_pattern'] = os.path.join(os.path.expandvars(config['data']['path']),
                                                  config['data']['file_pattern'])

    config['out_path'] = os.path.join('./inference', config['run_id'])  # edit

    inference(config)


if __name__ == '__main__':
    np.set_printoptions(suppress=True, formatter={'float_kind': '{:5.3}'.format})
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s, pid: %(process)d, %(levelname)-8s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        )
    main()
