"""
Inference script for the yolov3.yolov3_aleatoric class.

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
                    processed = self.process_batch(sess)

                    logging.info('Processed {} images.'.format((step - 1) * self.batch_size + processed))

                except tf.errors.OutOfRangeError:
                    break

            if self.worker_thread:
                self.worker_thread.join()
        return self

    def process_batch(self, sess):
        boxes, files = sess.run([self.nms, self.filename_tensor])

        if self.worker_thread:
            self.worker_thread.join()

        self.worker_thread = threading.Thread(target=self.write_to_disc, args=(boxes, files))
        self.worker_thread.start()
        return len(files)

    def write_to_disc(self, all_boxes, files):
        for batch, filename in enumerate(files):
            filename = filename[0].decode('utf-8')
            boxes = all_boxes[batch]
            self.write_ecp_json(boxes, filename)

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

def nms(all_boxes, model):
    def nms_op(boxes):
        # nms ignoring classes
        nms_indices = tf.image.non_max_suppression(boxes[:, :4], boxes[:, model.obj_idx], 1000)
        all_boxes = tf.gather(boxes, nms_indices, axis=0)
        all_boxes = tf.expand_dims(all_boxes, axis=0)

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

    body = lambda i, r: [i + 1, tf.concat([r, nms_op(all_boxes[i, ...])], axis=0)]

    r0 = nms_op(all_boxes[0, ...])  # do while
    i0 = tf.constant(1)  # start with 1!!!
    cond = lambda i, m: i < tf.shape(all_boxes)[0]
    ilast, result = tf.while_loop(cond, body, loop_vars=[i0, r0],
                                  shape_invariants=[i0.get_shape(), tf.TensorShape([None, None, all_boxes.shape[2]])])

    return result


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
        'x_var': float(bbox[4]),  # random value for models trained without aleatoric loss.
        'y_var': float(bbox[5]),  # random value for models trained without aleatoric loss.
        'w_var': float(bbox[6]),  # random value for models trained without aleatoric loss.
        'h_var': float(bbox[7]),  # random value for models trained without aleatoric loss.
        'total_var': float(bbox[8]),  # random value for models trained without aleatoric loss.
        'score': float(bbox[model.obj_idx]) * float(bbox[model.cls_start_idx + cls_idx]),
        'obj_entropy': float(bbox[model.obj_idx + 1]),
        'cls_scores': cls_scores,
        'cls_entropy': float(bbox[model.cls_start_idx + model.cls_cnt]),
        'layer_id': float(bbox[model.cls_start_idx + model.cls_cnt]),
        'prior_id': float(bbox[model.cls_start_idx + model.cls_cnt]),
        'identity': label_to_cls_name.get(cls, cls),
    }


def concat_bbox(net_out):
    bbox = None
    for det_layer in net_out:
        for prior in det_layer:
            batches, lw, lh, det_size = prior.shape.as_list()
            tmp = tf.reshape(prior, shape=[-1, lw * lh, det_size])
            if bbox is None:
                bbox = tmp
            else:
                bbox = tf.concat([bbox, tmp], axis=1)

    return bbox


# -----------------------------------------------------------------#
#                               main                               #
# -----------------------------------------------------------------#


def inference(config):
    assert not config['crop']
    logging.info(json.dumps(config, indent=4, default=lambda x: str(x)))

    logging.info('----- START -----')

    start = time.time()

    yolo = yolov3.yolov3_aleatoric(config)

    Inference(yolo, config).run()

    end = time.time()
    elapsed = int(end - start)
    logging.info('----- FINISHED in {:02d}:{:02d}:{:02d} -----'.format(elapsed // 3600,
                                                                       (elapsed // 60) % 60,
                                                                       elapsed % 60))


def main():
    config = {
        'checkpoint_path': './checkpoints',  # edit
        'run_id': 'pretraining',  # edit
        # 'step': 500000,  # edit
        'step': 'last',
        'full_img_size': [1024, 1920, 3],
        'cls_cnt': 2,  # edit
        'batch_size': 11,  # edit
        'cpu_thread_cnt': 24,  # edit
        'crop': False,
        'training': False,
        'aleatoric_loss': True,
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
