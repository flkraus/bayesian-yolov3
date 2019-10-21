import tensorflow as tf

from lib_yolo import data_augmentation, tfdata


def decode_img(encoded, shape):
    # decode image and scale to [0, 1)
    img = tf.image.decode_png(encoded, dtype=tf.uint8)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)  # convert to [0, 1)
    img.set_shape(shape)
    return img


def make_parse_fn(config):
    def parse_example(example):
        feature_map = {
            'image/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
            'image/height': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
            'image/width': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
            'image/object/class/label': tf.VarLenFeature(dtype=tf.int64),
            # 'image/object/class/text': tf.VarLenFeature(dtype=tf.string),
            'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
            # 'image/object/cnt': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
        }

        features = tf.parse_single_example(example, features=feature_map)

        img = decode_img(features['image/encoded'], config['full_img_size'])

        # assemble bbox
        xmin = tf.sparse_tensor_to_dense(features['image/object/bbox/xmin'], default_value=0)
        ymin = tf.sparse_tensor_to_dense(features['image/object/bbox/ymin'], default_value=0)
        xmax = tf.sparse_tensor_to_dense(features['image/object/bbox/xmax'], default_value=0)
        ymax = tf.sparse_tensor_to_dense(features['image/object/bbox/ymax'], default_value=0)
        bbox = tf.stack([ymin, xmin, ymax, xmax], axis=1)  # we use the standard tf bbox format

        label = tf.cast(tf.sparse_tensor_to_dense(features['image/object/class/label'], default_value=-1),
                        dtype=tf.int32)

        # Note regarding implicit background class:
        # The tensorflow object detection API enforces that the class labels start with 1.
        # The class 0 is reserved for an (implicit) background class.

        # yolo does not need a implicit background class.
        # To ensure compatibility with tf object detection API we support both: class ids starting at 1 or 0.
        implicit_background_class = config['implicit_background_class']
        if implicit_background_class:
            label = label - 1  # shift class 1 -> 0, 2 -> 1, etc...

        return img, bbox, label  # this is a mess

    return parse_example


def make_encode_bbox_fn(model_blueprint, config):
    def encode_bbox(img, bbox, label):
        gt = tfdata.encode_boxes(bbox, label, model_blueprint.det_layers, ign_thresh=config['ign_thresh'])
        return [img, *gt]

    return encode_bbox


def zero_center(img, *gt):
    img = 2 * (img - 0.5)  # [0, 1) -> [-1, 1)
    return [img, *gt]


def make_stack_same_img_fn_encoded(batch_size):
    def stack_same_image_encoded(img, *gt):
        img = tf.stack([img] * batch_size, axis=0)
        new_gt = []
        for gt_ in gt:
            new_gt.append({
                'loc': tf.stack([gt_['loc']] * batch_size, axis=0),
                'obj': tf.stack([gt_['obj']] * batch_size, axis=0),
                'cls': tf.stack([gt_['cls']] * batch_size, axis=0),
                'fpm': tf.stack([gt_['fpm']] * batch_size, axis=0),
                'ign': tf.stack([gt_['ign']] * batch_size, axis=0),
            })
        return [img, *new_gt]

    return stack_same_image_encoded


def make_stack_same_input_fn(batch_size):
    def stack_same_input(img, bbox, label):
        img = tf.stack([img] * batch_size, axis=0)
        bbox = tf.stack([bbox] * batch_size, axis=0)
        label = tf.stack([label] * batch_size, axis=0)
        return img, bbox, label

    return stack_same_input


def create_dataset(config, dataset_key):
    # in 1.9 list_files can shuffle directly...
    info = config[dataset_key]
    files = tf.data.Dataset.list_files(info['file_pattern']).shuffle(info['num_shards'])

    # cycle_length is important if the whole dataset does not fit into memory
    dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=2, block_length=1)

    dataset = dataset.map(make_parse_fn(config), num_parallel_calls=config['cpu_thread_cnt'])

    if info['cache']:
        dataset = dataset.cache()  # this fails if the dataset does not fit into memory
    return dataset


class TrainValDataset:
    def __init__(self, model_blueprint, config):
        encode_bbox_fn = make_encode_bbox_fn(model_blueprint, config)

        train_dataset = create_dataset(config, 'train')
        val_dataset = create_dataset(config, 'val')

        # process val dataset
        if config['crop']:
            val_dataset = val_dataset.map(config['val']['crop_fn'], num_parallel_calls=config['cpu_thread_cnt'])

        val_dataset = val_dataset.map(encode_bbox_fn, num_parallel_calls=config['cpu_thread_cnt'])

        val_dataset = val_dataset.shuffle(buffer_size=config['val']['shuffle_buffer_size'])
        val_dataset = val_dataset.repeat()
        val_dataset = val_dataset.prefetch(buffer_size=1)  # needed for val dataset?
        val_dataset = val_dataset.batch(batch_size=config['batch_size'])

        # process train dataset
        if config['crop']:
            train_dataset = train_dataset.map(config['train']['crop_fn'], num_parallel_calls=config['cpu_thread_cnt'])

        img_size = config['crop_img_size'] if config['crop'] else config['full_img_size']  # TODO move to fn?
        augmenter = data_augmentation.DataAugmenter(img_size)
        train_dataset = train_dataset.map(augmenter.augment, num_parallel_calls=config['cpu_thread_cnt'])

        train_dataset = train_dataset.map(encode_bbox_fn, num_parallel_calls=config['cpu_thread_cnt'])
        train_dataset = train_dataset.shuffle(buffer_size=config['train']['shuffle_buffer_size'])
        train_dataset = train_dataset.repeat()
        train_dataset = train_dataset.batch(batch_size=config['batch_size'])

        train_dataset = train_dataset.prefetch(buffer_size=1)

        self.__train_iterator = train_dataset.make_one_shot_iterator()
        self.__val_iterator = val_dataset.make_one_shot_iterator()

        # ---------------- #
        # public interface #
        # ---------------- #
        self.handle = tf.placeholder(tf.string, shape=[])
        self.iterator = tf.data.Iterator.from_string_handle(self.handle, train_dataset.output_types,
                                                            train_dataset.output_shapes)
        self.train_handle = None
        self.val_handle = None

    def init_dataset(self, sess):
        self.train_handle = sess.run(self.__train_iterator.string_handle())
        self.val_handle = sess.run(self.__val_iterator.string_handle())


class ValDataset:
    def __init__(self, config, map_fns=tuple(), dataset_key='data'):
        val_dataset = create_dataset(config, dataset_key)

        # process val dataset
        if config['crop']:
            val_dataset = val_dataset.map(config['val']['crop_fn'], num_parallel_calls=config['cpu_thread_cnt'])

        val_dataset = val_dataset.shuffle(buffer_size=config['val']['shuffle_buffer_size'])
        val_dataset = val_dataset.repeat()

        for map_fn in map_fns:
            val_dataset = val_dataset.map(map_fn, num_parallel_calls=24)

        val_dataset = val_dataset.map(make_stack_same_input_fn(batch_size=config['batch_size']),
                                      num_parallel_calls=config['cpu_thread_cnt'])
        val_dataset = val_dataset.prefetch(buffer_size=1)  # needed for val dataset?

        # ---------------- #
        # public interface #
        # ---------------- #
        self.handle = tf.placeholder(tf.string, shape=[])
        self.iterator = val_dataset.make_one_shot_iterator()


class TestingDataset:
    def __init__(self, config, config_key='data'):
        self.__config = config

        info = config[config_key]
        files = tf.data.Dataset.list_files(info['file_pattern'])
        # cycle_length is important if the whole dataset does not fit into memory
        dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=2, block_length=1)
        dataset = dataset.map(self.parse_example, num_parallel_calls=config['cpu_thread_cnt'])

        dataset = dataset.batch(batch_size=config['batch_size'])
        dataset = dataset.prefetch(buffer_size=1)  # needed for val dataset?

        # ---------------- #
        # public interface #
        # ---------------- #
        self.iterator = dataset.make_one_shot_iterator()

    def parse_example(self, example):
        feature_map = {
            'image/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
            'image/height': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
            'image/width': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
            'image/filename': tf.VarLenFeature(dtype=tf.string),
        }

        features = tf.parse_single_example(example, features=feature_map)

        img = decode_img(features['image/encoded'], self.__config['full_img_size'])

        filename = tf.sparse_tensor_to_dense(features['image/filename'], default_value='')
        return img, filename  # this is a mess
