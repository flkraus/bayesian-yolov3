import logging
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy.io
import tensorflow as tf


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


class ExampleCreator:
    def __init__(self, out_dir, dataset_name, label_to_text=None):
        self._out_dir = out_dir
        self._dataset_name = dataset_name

        # Create a single Session to run all image coding calls.
        self._sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 1}))

        # Initializes function that decodes RGB PNG data.
        self._decode_data = tf.placeholder(dtype=tf.string)
        self._decoded = tf.image.decode_png(self._decode_data, channels=3)

        self._encode_data = tf.placeholder(dtype=tf.uint8)
        self._encoded = tf.image.encode_png(self._encode_data)

        self.label_to_text = label_to_text or [
            'ignore',
            'pedestrian',
            'rider',
            'sitting',
            'unusual',
            'group',
        ]

    def get_shard_filename(self, shard, num_shards, split):
        shard_name = '{}-{}-{:05d}-of-{:05d}'.format(self._dataset_name, split, shard, num_shards)
        return os.path.join(self._out_dir, shard_name)

    def decode_png(self, img_data):
        img = self._sess.run(self._decoded, feed_dict={self._decode_data: img_data})
        assert len(img.shape) == 3
        assert img.shape[2] == 3
        return img

    def encode_png(self, img):
        assert len(img.shape) == 3
        assert img.shape[2] == 3
        return self._sess.run(self._encoded, feed_dict={self._encode_data: img})

    def load_img(self, path):
        ext = os.path.splitext(path)[1]
        if path.endswith('.pgm'):
            raise NotImplementedError('pgm not supported')
        if path.endswith('.png'):
            with tf.gfile.FastGFile(path, 'rb') as f:
                img_data = f.read()
            # seems a little bit stupid to first decode and then encode the image, but so what...
            return self.decode_png(img_data), ext[1:]
        else:
            raise NotImplementedError('unknown file format: {}'.format(ext))

    def create_example(self, img_path, annotations):
        img, format = self.load_img(img_path)
        img_height, img_width = img.shape[:2]
        assert img_height == 1024
        assert img_width == 2048
        encoded = self.encode_png(img)

        ymin, xmin, ymax, xmax, label, text, inst_id = [], [], [], [], [], [], []

        skipped_annotations = 0
        box_cnt = 0
        box_sizes = []
        for anno in annotations:
            anno = anno.astype(np.int64)  # this is important, otherwise overflows can occur

            class_label, x1, y1, w, h, instance_id, x1_vis, y1_vis, w_vis, h_vis = anno

            # we conform to the tf object detection API where 0 is reserved for the implicit background class
            # this ensures that tfrecord files which work with the object detection API also work with this framework
            if class_label == 2:
                # rider
                class_label = 2
            elif class_label in [0, 5]:
                # skip: ignore and group
                skipped_annotations += 1
                continue
            else:
                # pedestrian, sitting, unusual
                class_label = 1

            box_cnt += 1

            label_text = self.label_to_text[class_label]
            ymin.append(float(y1) / img_height)
            xmin.append(float(x1) / img_width)
            ymax.append(float(y1 + h) / img_height)
            xmax.append(float(x1 + w) / img_width)
            label.append(class_label)
            text.append(label_text.encode('utf8'))
            inst_id.append(instance_id)

            if 'group' not in label_text and 'ignore' not in label_text:
                # do not add group ore ignore boxes, we do not want these to affect the prior box calculation
                box_sizes.append((h, w))

        if skipped_annotations > 0:
            logging.debug(
                'Skipped {}/{} annotations for img {}'.format(skipped_annotations, len(annotations), img_path))

        feature_dict = {
            'image/height': int64_feature(img_height),
            'image/width': int64_feature(img_width),
            'image/filename': bytes_feature(img_path.encode('utf8')),
            'image/source_id': bytes_feature(img_path.encode('utf8')),
            'image/encoded': bytes_feature(encoded),
            'image/format': bytes_feature('png'.encode('utf8')),
            'image/object/bbox/xmin': float_list_feature(xmin),
            'image/object/bbox/xmax': float_list_feature(xmax),
            'image/object/bbox/ymin': float_list_feature(ymin),
            'image/object/bbox/ymax': float_list_feature(ymax),
            'image/object/class/text': bytes_list_feature(text),
            'image/object/class/label': int64_list_feature(label),
            'image/object/instance/id': int64_list_feature(inst_id),
            'image/object/cnt': int64_feature(box_cnt),
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature_dict))

        return example, skipped_annotations, box_sizes, (img_height, img_width)


def write_shard(args):
    shard, num_shards, split, data, img_dir, example_creator = args
    out_file = example_creator.get_shard_filename(shard, num_shards, split)

    writer = tf.python_io.TFRecordWriter(out_file)
    logging.info('Creating shard {}-{}/{}'.format(split, shard, num_shards))

    skipped_annotations = 0
    box_sizes = []
    img_sizes = set()
    cnt = 0
    for cnt, datum in enumerate(data, start=1):
        datum = datum[0][0]  # strange matlab file format
        city = str(datum[0][0])
        img_name = str(datum[1][0])
        annotations = datum[2]

        img_path = os.path.join(img_dir, city, img_name)

        example, skipped, sizes, img_size = example_creator.create_example(img_path, annotations)
        skipped_annotations += skipped
        box_sizes.extend(sizes)
        img_sizes.add(img_size)

        writer.write(example.SerializeToString())
        if cnt % 10 == 0:
            logging.info('Written {} examples for shard {}-{}/{}'.format(cnt, split, shard, num_shards))

    if skipped_annotations > 0:
        logging.info('Written {} examples for shard {}-{}/{}'.format(cnt, split, shard, num_shards))

    logging.info(
        'Finished shard {}-{}/{}: {} examples written and {} annotations skipped'.format(split, shard, num_shards, cnt,
                                                                                         skipped_annotations))
    return box_sizes, split, img_sizes


def create_jobs(split, shuffle, annotations, img_dir, num_shards, example_creator):
    if shuffle:
        np.random.shuffle(annotations)

    # split into roughly even sized pieces
    k, m = divmod(len(annotations), num_shards)
    shards = [annotations[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(num_shards)]

    # check if we didn't f@#! it up
    total_length = 0
    for shard in shards:
        total_length += shard.shape[0]
    assert total_length == len(annotations)

    # create and run jobs
    jobs = [(shard_id + 1, num_shards, split, data, img_dir, example_creator) for shard_id, data in enumerate(shards)]
    return jobs


def create_dirs(dirs):
    for path in dirs:
        try:
            os.makedirs(path)
        except OSError:
            assert os.path.isdir(path), '{} exists but is not a directory'.format(path)


def process_dataset(out_dir, dataset_name, anno_dir, img_dir, train_shards, val_shards, shuffle):
    out_dir = os.path.expandvars(out_dir)
    img_dir = os.path.expandvars(img_dir)
    anno_dir = os.path.expandvars(anno_dir)

    create_dirs([out_dir])

    if shuffle:
        with open(os.path.join(out_dir, '{}-np_random_state'.format(dataset_name)), 'wb') as f:
            pickle.dump(np.random.get_state(), f)

    # prepare train and val splits
    train_anno_path = os.path.join(anno_dir, 'annotations', 'anno_train.mat')
    val_anno_path = os.path.join(anno_dir, 'annotations', 'anno_val.mat')

    train_img_dir_ = os.path.join(img_dir, 'leftImg8bit_trainvaltest', 'leftImg8bit', 'train')
    val_img_dir = os.path.join(img_dir, 'leftImg8bit_trainvaltest', 'leftImg8bit', 'val')

    train_anno = scipy.io.loadmat(train_anno_path)['anno_train_aligned'][0]  # citypersons data format
    val_anno = scipy.io.loadmat(val_anno_path)['anno_val_aligned'][0]  # citypersons data format

    # object which does all the hard work
    example_creator = ExampleCreator(out_dir, dataset_name)

    # Process each split in a different thread
    train_jobs = create_jobs('train', shuffle, train_anno, train_img_dir_, train_shards, example_creator)
    val_jobs = create_jobs('val', shuffle, val_anno, val_img_dir, val_shards, example_creator)

    jobs = train_jobs + val_jobs

    with ThreadPoolExecutor() as executor:
        result = executor.map(write_shard, jobs,
                              chunksize=1)  # chunksize=1 is important, since our jobs are long running

    box_sizes = []
    img_sizes = set()
    for sizes, split, img_sizes_ in result:
        img_sizes.update(img_sizes_)
        if split == 'train':
            box_sizes.extend(sizes)

    if len(img_sizes) > 1:
        logging.error('Different image sizes detected: {}'.format(img_sizes))

    box_sizes = np.array(box_sizes, np.float64)
    np.save(os.path.join(out_dir, '{}-train-box_sizes'.format(dataset_name)), box_sizes)
    np.save(os.path.join(out_dir, '{}-img_size_height_width'.format(dataset_name)), list(img_sizes)[0])


def main():
    config = {
        # Place to search for the created files.
        'out_dir': '$HOME/data/citypersons/tfrecords_test',

        # Name of the dataset, used to create the tfrecord files.
        'dataset_name': 'citypersons',

        # Base directory which contains the citypersons annotations.
        'anno_dir': '$HOME/data/citypersons',  # edit

        # Base directory which contains the cityscapes images.
        'img_dir': '$HOME/data/cityscapes',

        # Number of training and validation shards.
        'train_shards': 3,
        'val_shards': 1,

        # Shuffle the data before writing it to tfrecord files.
        'shuffle': True,
    }

    logging.info('Saving results to {}'.format(config['out_dir']))
    logging.info('----- START -----')
    start = time.time()

    process_dataset(**config)

    end = time.time()
    elapsed = int(end - start)
    logging.info('----- FINISHED in {:02d}:{:02d}:{:02d} -----'.format(elapsed // 3600,
                                                                       (elapsed // 60) % 60,
                                                                       elapsed % 60))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,  # edit change to DEBUG for more detailed output
                        format='%(asctime)s, %(levelname)-8s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        )

    main()
