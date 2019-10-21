import glob
import logging
import multiprocessing
import os
import time

import matplotlib.cm
import numpy as np
import tensorflow as tf
from PIL import Image

from lib_yolo import yolov3


def colorize(img, vmin=None, vmax=None, cmap='plasma'):
    # normalize
    vmin = tf.reduce_min(img) if vmin is None else vmin
    vmax = tf.contrib.distributions.percentile(img, 99.) if vmax is None else vmax
    img = (img - vmin) / (vmax - vmin)

    img = tf.squeeze(img, axis=[-1])

    # quantize
    indices = tf.clip_by_value(tf.to_int32(tf.round(img * 255)), 0, 255)

    # gather
    cm = matplotlib.cm.get_cmap(cmap if cmap is not None else 'gray')
    colors = tf.constant(cm.colors, dtype=tf.float32)
    img = tf.gather(colors, indices)

    return img


def color_map(img, uncertainty, stride, vmin, vmax, alpha=0.7):
    uncertainty = colorize(uncertainty, vmin, vmax)
    uncertainty = tf.expand_dims(uncertainty, axis=0)
    shape = uncertainty.shape
    uncertainty = tf.image.resize_nearest_neighbor(uncertainty, size=(shape[1] * stride, shape[2] * stride))

    blended = alpha * img + (1 - alpha) * uncertainty
    tf.squeeze(blended, axis=0)
    blended = blended[0, ...]

    blended = tf.image.convert_image_dtype(blended, dtype=tf.uint8)  # convert to [0, 255]

    return blended


class Inference:
    def __init__(self, yolo, config):
        self.batch_size = config['batch_size']
        self.img_size = yolo.img_size
        self.img_tensor = tf.placeholder(tf.float32, shape=(1, *self.img_size))
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

        self.config = config
        self.worker_thread = None

        assert config['inference_mode']
        self.model = yolo.init_model(inputs=self.img_tensor, training=False).get_model()

        self.grids = []
        stats = [None] * 9

        ucty_idx = config.get('ucty_idx', -1)
        uncertainty_key = config['uncertainty_key']

        # stride 32
        l = self.model.det_layers[0]
        if 'obj' in uncertainty_key or 'cls' in uncertainty_key:
            uncertainty = l.det[uncertainty_key]
        elif 'epi' in uncertainty_key:
            uncertainty = l.det[uncertainty_key][..., ucty_idx, ucty_idx]
        else:
            uncertainty = l.det[uncertainty_key][..., ucty_idx]
        lh, lw, box_cnt = uncertainty.shape.as_list()
        uncertainty = tf.split(uncertainty, [1] * box_cnt, axis=-1)

        self.grids.append(
            color_map(self.img_tensor, uncertainty[0], l.downsample, 0, stats[0]))
        self.grids.append(
            color_map(self.img_tensor, uncertainty[1], l.downsample, 0, stats[1]))
        self.grids.append(
            color_map(self.img_tensor, uncertainty[2], l.downsample, 0, stats[2]))

        # stride 16
        l = self.model.det_layers[1]
        if 'obj' in uncertainty_key or 'cls' in uncertainty_key:
            uncertainty = l.det[uncertainty_key]
        elif 'epi' in uncertainty_key:
            uncertainty = l.det[uncertainty_key][..., ucty_idx, ucty_idx]
        else:
            uncertainty = l.det[uncertainty_key][..., ucty_idx]
        lh, lw, box_cnt = uncertainty.shape.as_list()
        uncertainty = tf.split(uncertainty, [1] * box_cnt, axis=-1)

        self.grids.append(
            color_map(self.img_tensor, uncertainty[0], l.downsample, 0, stats[3]))
        self.grids.append(
            color_map(self.img_tensor, uncertainty[1], l.downsample, 0, stats[4]))
        self.grids.append(
            color_map(self.img_tensor, uncertainty[2], l.downsample, 0, stats[5]))

        # stride 8
        l = self.model.det_layers[2]
        if 'obj' in uncertainty_key or 'cls' in uncertainty_key:
            uncertainty = l.det[uncertainty_key]
        elif 'epi' in uncertainty_key:
            uncertainty = l.det[uncertainty_key][..., ucty_idx, ucty_idx]
        else:
            uncertainty = l.det[uncertainty_key][..., ucty_idx]
        lh, lw, box_cnt = uncertainty.shape.as_list()
        uncertainty = tf.split(uncertainty, [1] * box_cnt, axis=-1)

        self.grids.append(
            color_map(self.img_tensor, uncertainty[0], l.downsample, 0, stats[6]))
        self.grids.append(
            color_map(self.img_tensor, uncertainty[1], l.downsample, 0, stats[7]))
        self.grids.append(
            color_map(self.img_tensor, uncertainty[2], l.downsample, 0, stats[8]))

        self.sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 1}))
        tf.train.Saver().restore(self.sess, self.checkpoint)

    def load_img(self, filename):
        img = Image.open(filename)
        img = np.array(img)
        img = img.astype(np.float32)

        if self.config['crop']:
            y = (img.shape[0] - self.img_size[0]) // 2
            x = (img.shape[1] - self.img_size[1]) // 2
            img = img[y:y + self.img_size[0], x:x + self.img_size[1], :]

        img = np.expand_dims(img, axis=0)
        img /= 255.
        return img

    def make_color_map(self, filename, config):
        img_data = self.load_img(filename)
        grids, = self.sess.run([self.grids], feed_dict={self.img_tensor: img_data})
        img_name = os.path.basename(filename)
        save_uncertainty_maps(grids, img_name, config)


def save_uncertainty_maps(grids, file_name, config):
    file_name = os.path.basename(file_name)
    for idx, img in enumerate(grids):
        result = Image.fromarray(img)
        path = os.path.join(config['out_path'],
                            '{}_prior{}_{}.png'.format(os.path.splitext(file_name)[0], idx, config['ucty']))
        result.save(path)


def worker(files, config):
    os.makedirs(config['out_path'], exist_ok=True)
    yolo = yolov3.bayesian_yolov3_aleatoric(config)
    inference = Inference(yolo, config)

    logging.info('Processing: {}'.format(config['ucty']))
    for file in files:
        logging.info('Processing file: {}'.format(file))
        inference.make_color_map(file, config)
    logging.info('Finished: {}'.format(config['ucty']))


def do_it(files, config):
    for uncertainty_key in ['epi_covar_loc', 'ale_var_loc']:
        for ucty_idx in range(4):
            if 'epi' in uncertainty_key:
                ucty_type = 'epi'
            else:
                ucty_type = 'ale'

            mapping = ['x', 'y', 'w', 'h']

            config['ucty'] = ucty_type + '_' + mapping[ucty_idx]
            config['ucty_idx'] = ucty_idx
            config['uncertainty_key'] = uncertainty_key

            p = multiprocessing.Process(target=worker, args=(files, config))
            p.start()
            p.join()

    for uncertainty_key in ['cls_mutual_info', 'obj_mean', 'obj_mutual_info']:
        config['uncertainty_key'] = uncertainty_key
        config['ucty'] = uncertainty_key

        p = multiprocessing.Process(target=worker, args=(files, config))
        p.start()
        p.join()


def main():
    config = {
        'checkpoint_path': './checkpoints/',
        'run_id': 'epi_ale',  # edit
        'step': 'last',  # edit, int or 'last'
        'crop_img_size': [768, 1440, 3],
        'full_img_size': [1024, 1920, 3],  # edit if not ecp
        'cls_cnt': 2,
        'batch_size': 1,
        'T': 30,
        'inference_mode': True,
        'cpu_thread_cnt': 10,
        'freeze_darknet53': False,  # actual value irrelevant
        'crop': False,  # edit
        'training': False,
        'aleatoric_loss': True,  # actual value irrelevant
        'priors': yolov3.ECP_9_PRIORS,  # actual value irrelevant
        'out_path': './uncertainty_visualization',  # edit
    }

    # NOTE: only works for bayesian_yolov3_aleatoric clss
    assert config['batch_size'] == 1
    assert config['inference_mode']

    files = glob.glob('./test_images/*')  # edit

    logging.info('----- START -----')
    start = time.time()

    do_it(files, config)

    end = time.time()
    elapsed = int(end - start)
    logging.info('----- FINISHED in {:02d}:{:02d}:{:02d} -----'.format(elapsed // 3600,
                                                                       (elapsed // 60) % 60,
                                                                       elapsed % 60))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s, pid: %(process)d, %(levelname)-8s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        )
    main()
