import tensorflow as tf


class DataAugmenter:
    def __init__(self, img_size):
        self.__img_size = img_size
        self.__ones_h_w_c = tf.ones(shape=img_size, dtype=tf.float32)
        self.__ones_h_w = tf.ones(shape=img_size[:2], dtype=tf.float32)
        self.__zeros_h_w_c = tf.zeros(shape=img_size, dtype=tf.float32)
        self.__zeros_h_w = tf.zeros(shape=img_size[:2], dtype=tf.float32)

        self.__int = [tf.constant(num, dtype=tf.int32) for num in range(10)]
        self.__rand_0_1 = {
            'shape': (),
            'minval': 0,
            'maxval': 1,
            'dtype': tf.float32
        }

    def augment(self, img, bbox, label):
        # flip 50%
        img, bbox = tf.cond(tf.random_uniform(**self.__rand_0_1) < 0.5, true_fn=lambda: self.flip_lr(img, bbox),
                            false_fn=lambda: (img, bbox))

        # blur 5% of images
        img = tf.cond(tf.random_uniform(**self.__rand_0_1) < 0.05, true_fn=lambda: self.blur(img), false_fn=lambda: img)

        # add color augmentation (hue, brightness, ...) to 5% of images (additional to blur)
        img = tf.cond(tf.random_uniform(**self.__rand_0_1) < 0.05, true_fn=lambda: self.color_augmentations(img),
                      false_fn=lambda: img)

        # add noise augmentation (salt&pepper, gaussian noise, ...) to 5% of images (additional to blur and color augm)
        img = tf.cond(tf.random_uniform(**self.__rand_0_1) < 0.05, true_fn=lambda: self.noise_augmentations(img),
                      false_fn=lambda: img)

        return img, bbox, label

    def color_augmentations(self, img):
        rand_args = {
            'shape': (),
            'minval': 0,
            'maxval': 3,
            'dtype': tf.int32
        }
        choice = tf.random_uniform(**rand_args)

        img = tf.cond(tf.equal(choice, tf.constant(0, dtype=tf.int32)),
                      true_fn=lambda: tf.image.random_saturation(img, 0.5, 1.5), false_fn=lambda: img)
        img = tf.cond(tf.equal(choice, tf.constant(1, dtype=tf.int32)),
                      true_fn=lambda: tf.image.random_brightness(img, 0.2), false_fn=lambda: img)
        img = tf.cond(tf.equal(choice, tf.constant(2, dtype=tf.int32)),
                      true_fn=lambda: tf.image.random_hue(img, 0.2), false_fn=lambda: img)

        return img

    def noise_augmentations(self, img):
        rand_args = {
            'shape': (),
            'minval': 0,
            'maxval': 3,
            'dtype': tf.int32
        }
        choice = tf.random_uniform(**rand_args)

        img = tf.cond(tf.equal(choice, tf.constant(0, dtype=tf.int32)),
                      true_fn=lambda: self.colored_salt_n_pepper(img), false_fn=lambda: img)
        img = tf.cond(tf.equal(choice, tf.constant(1, dtype=tf.int32)),
                      true_fn=lambda: self.salt_n_pepper(img), false_fn=lambda: img)
        img = tf.cond(tf.equal(choice, tf.constant(2, dtype=tf.int32)),
                      true_fn=lambda: self.additive_gaussian_noise(img), false_fn=lambda: img)

        return img

    def flip_lr(self, img, bbox):
        img = tf.image.flip_left_right(img)

        ymin, xmin, ymax, xmax = tf.split(value=bbox, num_or_size_splits=4, axis=1)
        flipped_xmin = 1.0 - xmax
        flipped_xmax = 1.0 - xmin
        bbox = tf.concat([ymin, flipped_xmin, ymax, flipped_xmax], axis=1)

        return img, bbox

    def colored_salt_n_pepper(self, img):
        # season each channel individually
        salt_mask = tf.random_uniform(shape=self.__img_size, minval=0, maxval=1)
        pepper_mask = tf.random_uniform(shape=self.__img_size, minval=0, maxval=1)
        amount = tf.random_uniform(shape=(), minval=0.0005, maxval=0.008)
        img = tf.where(tf.less(salt_mask, amount), self.__ones_h_w_c, img)
        img = tf.where(tf.less(pepper_mask, amount), self.__zeros_h_w_c, img)

        return img

    def salt_n_pepper(self, img):
        # season all channels at the same time
        size = self.__img_size[:2]
        salt_mask = tf.random_uniform(shape=size, minval=0, maxval=1)
        pepper_mask = tf.random_uniform(shape=size, minval=0, maxval=1)
        amount = tf.random_uniform(shape=(), minval=0.0005, maxval=0.008)

        salt = tf.where(tf.less(salt_mask, amount), self.__ones_h_w, self.__zeros_h_w)
        pepper = tf.where(tf.less(pepper_mask, amount), -self.__ones_h_w, self.__zeros_h_w)  # note the minus sign!

        # if a pepper corn and a salt cristal fall on the same spot they vanish both. Magic!
        salt_n_pepper = salt + pepper
        img = img + tf.expand_dims(salt_n_pepper, 2)  # season your image
        img = tf.clip_by_value(img, clip_value_min=0, clip_value_max=1)  # remove excess seasoning
        return img

    def blur(self, img):
        kernel_size = tf.random_uniform(shape=(), minval=2, maxval=4, dtype=tf.int32)  # 2 or 3
        kernel = tf.ones(shape=tf.stack((kernel_size, kernel_size, 1, 1)))
        kernel_size = tf.cast(kernel_size, tf.float32)
        kernel = kernel / (kernel_size * kernel_size)

        r, g, b = tf.unstack(tf.expand_dims(img, 0), num=3, axis=3)
        r = tf.expand_dims(r, 3)
        g = tf.expand_dims(g, 3)
        b = tf.expand_dims(b, 3)

        r = tf.nn.conv2d(r, filter=kernel, strides=[1, 1, 1, 1], padding='SAME')
        g = tf.nn.conv2d(g, filter=kernel, strides=[1, 1, 1, 1], padding='SAME')
        b = tf.nn.conv2d(b, filter=kernel, strides=[1, 1, 1, 1], padding='SAME')

        img = tf.concat([r, g, b], axis=3)
        return tf.squeeze(img, axis=0)

    def additive_gaussian_noise(self, img):
        stddev = tf.random_uniform(shape=(), minval=0.001, maxval=0.05, dtype=tf.float32)

        noise = tf.random_normal(shape=self.__img_size, mean=0.0, stddev=stddev)
        img = img + noise
        return img


class ImageCropper:
    def __init__(self, config):
        self.config = config
        self.crop_height = config['crop_img_size'][0]
        self.crop_width = config['crop_img_size'][1]
        self.full_height = config['full_img_size'][0]
        self.full_width = config['full_img_size'][1]

        aspect_ratio_full_img = self.full_width / float(self.full_height)
        aspect_ratio_crop_img = self.crop_width / float(self.crop_height)

        # comparing two floats with "==" is always a smart idea (10/10, would do it again).
        assert aspect_ratio_full_img == aspect_ratio_crop_img, 'invalid crop aspect ratio, must be same as full image'

    def random_crop_and_sometimes_rescale(self, img, bbox, label):
        # randomly rescales the crop area 33% of the time, else normal random crop
        img, bbox, label = tf.cond(tf.random_uniform(shape=(), minval=0.0, maxval=1.0, dtype=tf.float32) < 0.33,
                                   true_fn=lambda: self.random_crop_with_rescale(img, bbox, label),
                                   false_fn=lambda: self.random_crop(img, bbox, label))
        return img, bbox, label

    def random_crop_with_rescale(self, img, bbox, label):
        """
        First select a crop of random size, then resize to desired crop_size.
        """
        with tf.name_scope('random_crop_and_rescale_data'):
            scale = tf.clip_by_value(tf.random_normal(shape=(), mean=0, stddev=0.5, dtype=tf.float32),
                                     clip_value_min=-0.7, clip_value_max=0.7)
            crop_height = tf.cast(tf.minimum((1 + scale) * self.crop_height, self.full_height), tf.int32)
            crop_width = tf.cast(tf.minimum((1 + scale) * self.crop_width, self.full_width), tf.int32)

            # prefer crops in the middle of the image in y direction (use normal dist to select y)
            y_maxval = tf.cast(self.full_height - crop_height, tf.float32)
            y_mean = y_maxval / 2.
            stddev_y = y_maxval / 4.
            y_min_ind = tf.random_normal(shape=(), mean=y_mean, stddev=stddev_y, dtype=tf.float32)
            y_min_ind = tf.clip_by_value(y_min_ind, 0, y_maxval)
            y_min_ind = tf.cast(y_min_ind, tf.int32)

            # crops are uniformly distributed in x direction
            x_min_ind = tf.random_uniform(shape=(), minval=0, maxval=self.full_width - crop_width + 1,
                                          dtype=tf.int32)

            y_min = tf.cast(y_min_ind, tf.float32) / self.full_height
            x_min = tf.cast(x_min_ind, tf.float32) / self.full_width
            y_max = y_min + (tf.cast(crop_height, tf.float32) / float(self.full_height))
            x_max = x_min + (tf.cast(crop_width, tf.float32) / float(self.full_width))

            img = tf.image.crop_to_bounding_box(img, y_min_ind, x_min_ind, crop_height, crop_width)
            bbox, label = crop_boxes(bbox, label, y_min, x_min, y_max, x_max)

            img = tf.image.resize_images(img, [self.crop_height, self.crop_width])

        return img, bbox, label

    def random_crop(self, img, bbox, label):
        with tf.name_scope('random_crop_data'):
            # prefer crops in the middle of the image in y direction (use normal dist to select y)
            y_maxval = self.full_height - self.crop_height
            y_mean = y_maxval / 2
            stddev_y = y_maxval / 4
            y_min_ind = tf.random_normal(shape=(), mean=y_mean, stddev=stddev_y, dtype=tf.float32)
            y_min_ind = tf.clip_by_value(y_min_ind, 0, y_maxval)
            y_min_ind = tf.cast(y_min_ind, tf.int32)

            # crops are uniformly distributed in x direction
            x_min_ind = tf.random_uniform(shape=(), minval=0, maxval=self.full_width - self.crop_width + 1,
                                          dtype=tf.int32)

            y_min = tf.cast(y_min_ind, tf.float32) / self.full_height
            x_min = tf.cast(x_min_ind, tf.float32) / self.full_width
            y_max = y_min + (self.crop_height / float(self.full_height))
            x_max = x_min + (self.crop_width / float(self.full_width))

            img = tf.image.crop_to_bounding_box(img, y_min_ind, x_min_ind, self.crop_height, self.crop_width)
            bbox, label = crop_boxes(bbox, label, y_min, x_min, y_max, x_max)

        return img, bbox, label

    def center_crop(self, img, bbox, label):
        with tf.name_scope('center_crop_data'):
            y_min_ind = (self.full_height - self.crop_height) // 2
            x_min_ind = (self.full_width - self.crop_width) // 2

            y_min = tf.cast(y_min_ind, tf.float32) / self.full_height
            x_min = tf.cast(x_min_ind, tf.float32) / self.full_width
            y_max = y_min + (self.crop_height / float(self.full_height))
            x_max = x_min + (self.crop_width / float(self.full_width))

            img = tf.image.crop_to_bounding_box(img, y_min_ind, x_min_ind, self.crop_height, self.crop_width)
            bbox, label = crop_boxes(bbox, label, y_min, x_min, y_max, x_max)

        return img, bbox, label


def crop_boxes(boxes, labels, crop_y_min, crop_x_min, crop_y_max, crop_x_max, thresh=0.25):
    with tf.name_scope('crop_boxes'):
        y_min, x_min, y_max, x_max = tf.split(boxes, num_or_size_splits=4, axis=1)
        areas = tf.squeeze((y_max - y_min) * (x_max - x_min), [1])
        y_min_clipped = tf.maximum(tf.minimum(y_min, crop_y_max), crop_y_min)
        y_max_clipped = tf.maximum(tf.minimum(y_max, crop_y_max), crop_y_min)
        x_min_clipped = tf.maximum(tf.minimum(x_min, crop_x_max), crop_x_min)
        x_max_clipped = tf.maximum(tf.minimum(x_max, crop_x_max), crop_x_min)
        clipped = tf.concat([(y_min_clipped - crop_y_min) / (crop_y_max - crop_y_min),
                             (x_min_clipped - crop_x_min) / (crop_x_max - crop_x_min),
                             (y_max_clipped - crop_y_min) / (crop_y_max - crop_y_min),
                             (x_max_clipped - crop_x_min) / (crop_x_max - crop_x_min)], axis=1)

        areas_clipped = tf.squeeze((y_max_clipped - y_min_clipped) * (x_max_clipped - x_min_clipped), [1])

        # remove all boxes which are less than xx% of their original area (thresh = 0.25 ~ 25%)
        nonzero_area_indices = tf.cast(tf.reshape(tf.where(tf.greater(areas_clipped / areas, thresh)), [-1]), tf.int32)
        clipped = tf.gather(clipped, nonzero_area_indices)
        clipped_labels = tf.gather(labels, nonzero_area_indices)

        return clipped, clipped_labels


def box_area(boxes):
    y_min, x_min, y_max, x_max = tf.split(value=boxes, num_or_size_splits=4, axis=1)
    return tf.squeeze((y_max - y_min) * (x_max - x_min), [1])
