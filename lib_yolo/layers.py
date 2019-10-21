import tensorflow as tf


def flatten(*tensors):
    flattened = []
    for t in tensors:
        flattened.append(tf.layers.flatten(t))
    return flattened


def split_detection(inputs, boxes_per_cell, cls_cnt):  # vanilla YOLOv3
    # split the prediction feature map into localization, objectness prediction and class prediction

    loc, obj, cls = [], [], []

    # split feature map to get the predictions for different priors
    split_boxes = tf.split(inputs, boxes_per_cell, axis=-1)

    for i, dets in enumerate(split_boxes):
        # for this to work we need 'channels_last' (axis=-1)
        loc_, obj_, cls_ = tf.split(dets, [4, 1, cls_cnt], axis=-1)
        loc.append(loc_)
        obj.append(obj_)
        cls.append(cls_)

    with tf.name_scope('loc'):
        loc = tf.stack(loc, axis=-2)  # shape=(b, h, w, boxes_per_cell, loc_size), where loc_size=4 (x, y, w, h)
    with tf.name_scope('obj'):
        obj = tf.stack(obj, axis=-2)
        obj = tf.squeeze(obj, axis=-1)  # shape=(b, h, w, boxes_per_cell)
    with tf.name_scope('cls'):
        cls = tf.stack(cls, axis=-2)  # shape=(b, h, w, boxes_per_cell, cls_cnt)

    return {
        'loc': loc,
        'obj': obj,
        'cls': cls,
    }


def split_detection_aleatoric(inputs, boxes_per_cell, cls_cnt):
    # split the prediction feature map into localization, objectness prediction and class prediction

    loc, log_loc_var, obj, log_obj_stddev, cls, log_cls_stddev = [], [], [], [], [], []

    # split feature map to get the predictions for different priors
    split_boxes = tf.split(inputs, boxes_per_cell, axis=-1)

    for i, dets in enumerate(split_boxes):
        # for this to work we need 'channels_last' (axis=-1)
        loc_, log_loc_var_, obj_, log_obj_stddev_, cls_, log_cls_stddev_ = tf.split(dets,
                                                                                    [4, 4, 1, 1, cls_cnt, cls_cnt],
                                                                                    axis=-1)
        loc.append(loc_)
        log_loc_var.append(log_loc_var_)
        obj.append(obj_)
        log_obj_stddev.append(log_obj_stddev_)
        cls.append(cls_)
        log_cls_stddev.append(log_cls_stddev_)

    with tf.name_scope('loc'):
        loc = tf.stack(loc, axis=-2)  # shape=(b, h, w, boxes_per_cell, loc_size), where loc_size=4 (x, y, w, h)
    with tf.name_scope('log_loc_var'):
        log_loc_var = tf.stack(log_loc_var,
                               axis=-2)  # shape=(b, h, w, boxes_per_cell, loc_size), where loc_size=4 (x, y, w, h)
    with tf.name_scope('obj'):
        obj = tf.stack(obj, axis=-2)
        obj = tf.squeeze(obj, axis=-1)  # shape=(b, h, w, boxes_per_cell)
    with tf.name_scope('log_obj_stddev'):
        log_obj_stddev = tf.stack(log_obj_stddev, axis=-2)
        log_obj_stddev = tf.squeeze(log_obj_stddev, axis=-1)  # shape=(b, h, w, boxes_per_cell)
    with tf.name_scope('cls'):
        cls = tf.stack(cls, axis=-2)  # shape=(b, h, w, boxes_per_cell, cls_cnt)
    with tf.name_scope('log_cls_stddev'):
        log_cls_stddev = tf.stack(log_cls_stddev, axis=-2)  # shape=(b, h, w, boxes_per_cell, cls_cnt)

    return {
        'loc': loc,
        'log_loc_var': log_loc_var,
        'obj': obj,
        'log_obj_stddev': log_obj_stddev,
        'cls': cls,
        'log_cls_stddev': log_cls_stddev,
    }


def aleatoric_obj_loss(det, gt):  # aleatoric classification loss attenuation of Alex Kendall, not active
    T = 42  # this is completely random

    expected_value = tf.zeros_like(det['obj'])
    obj_stddev = tf.exp(tf.clip_by_value(det['log_obj_stddev'], -40, 40))  # this guarantees positive values
    for i in range(T):
        eps = tf.random_normal(tf.shape(det['obj']), mean=0.0, stddev=1.0)
        x = det['obj'] + (obj_stddev * eps)  # sample logits
        s = tf.sigmoid(x)
        p = tf.where(gt['obj'] > 0.5, s, 1 - s)  # sigmoid probability of true class
        expected_value = expected_value + p
    expected_value = expected_value / float(T)

    log_loss = - tf.log(expected_value)  # don't forget the minus sign
    return log_loss


def aleatoric_cls_loss(det, gt):  # aleatoric classification loss attenuation of Alex Kendall, not active
    T = 42  # this is completely random

    cls_cnt = tf.shape(det['cls'])[-1]
    gt_one_hot = tf.one_hot(gt['cls'], cls_cnt)

    # [sic] we want to use shape of gt['cls'] not det['cls']!
    expected_value = tf.zeros_like(gt['cls'], dtype=tf.float32)
    cls_stddev = tf.exp(tf.clip_by_value(det['log_cls_stddev'], -40, 40))  # this guarantees positive values
    for i in range(T):
        eps = tf.random_normal(tf.shape(det['cls']), mean=0.0, stddev=1.0)
        x = det['cls'] + (cls_stddev * eps)  # sample logits
        s = tf.nn.softmax(x)
        # calculate softmax probability of true class (this is a little bit hacky...)
        p = tf.reduce_sum(s * gt_one_hot, axis=-1)
        expected_value = expected_value + p
    expected_value = expected_value / float(T)

    log_loss = - tf.log(expected_value)  # don't forget the minus sign
    return log_loss


def loss_tf(det, gt, aleatoric_loss=False):
    """
    :param det: dict:
            ['loc'] tensor with shape=(b, h, w, boxes_per_cell, loc_size), where loc_size=4 (x, y, w, h)
            ['log_loc_var'] same shape as 'loc' but only guaranteed to be present if aleatoric_loss=True
            ['obj'] tensor with shape=(b, h, w, boxes_per_cell)
            ['cls'] tensor with shape=(b, h, w, boxes_per_cell, cls_cnt)
    :param gt: dict:
            ['loc'] tensor with shape=(b, h, w, boxes_per_cell, loc_size), where loc_size=4 (x, y, w, h)
            ['obj'] tensor with shape=(b, h, w, boxes_per_cell)
            ['cls'] tensor with shape=(b, h, w, boxes_per_cell, cls_cnt)

    :param aleatoric_loss: If the aleatoric localization loss should be added
    :return:
    """

    with tf.name_scope('batch_size'):
        batch_size = tf.cast(tf.shape(det['loc'])[0], dtype=tf.float32)

    with tf.name_scope('loss'):
        with tf.name_scope('localization'):
            loc_loss = (gt['loc'] - det['loc'])
            loc_loss = loc_loss ** 2

            if aleatoric_loss:
                log_loc_var = tf.clip_by_value(det['log_loc_var'], clip_value_min=-40, clip_value_max=40)
                loc_loss = loc_loss * tf.exp(-log_loc_var)
                loc_loss = loc_loss + log_loc_var

            loc_loss = loc_loss * tf.expand_dims(gt['obj'], axis=-1)  # only apply loss to cells if there is an object
            loc_loss_all = tf.reduce_sum(loc_loss) / (2 * batch_size)
            tf.summary.scalar('value', loc_loss_all)
            tf.losses.add_loss(loc_loss_all)

        with tf.name_scope('objectness'):
            # if aleatoric_loss:
            #     obj_loss = aleatoric_obj_loss(det, gt)
            # else:
            obj_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=gt['obj'], logits=det['obj'])

            obj_loss = obj_loss * gt['ign']
            obj_loss_all = tf.reduce_sum(obj_loss) / batch_size
            tf.summary.scalar('value', obj_loss_all)
            tf.losses.add_loss(obj_loss_all)

        with tf.name_scope('cls_x_entropy'):
            # if aleatoric_loss:
            #     cls_loss = aleatoric_cls_loss(det, gt)
            # else:
            cls_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gt['cls'], logits=det['cls'])

            cls_loss = cls_loss * gt['obj']  # only apply loss to cells if there is an object
            cls_loss_all = tf.reduce_sum(cls_loss) / batch_size
            tf.summary.scalar('value', cls_loss_all)
            tf.losses.add_loss(cls_loss_all)

        tf.summary.scalar('total', loc_loss_all + obj_loss_all + cls_loss_all)
    loss = {
        'loc': loc_loss_all,
        'obj': obj_loss_all,
        'cls': cls_loss_all,
    }
    return loss


def decode_bbox_standard(det, priors):
    """
    Converts center x, center y, width and height values to coordinates of top left and bottom right points.

    :param predictions: outputs of YOLO v3 detector of shape (?, h, w, boxes_per_cell * (cls_cnt + 5))
    :return: converted detections of same shape as input
    """

    batches, lh, lw, box_cnt, cls_cnt = det['cls'].shape.as_list()
    assert box_cnt == len(priors)

    obj_ = tf.sigmoid(det['obj'])
    cls_ = tf.nn.softmax(det['cls'])

    loc_split = tf.split(det['loc'], [1] * box_cnt, axis=-2)
    obj_split = tf.split(obj_, [1] * box_cnt, axis=-1)
    cls_split = tf.split(cls_, [1] * box_cnt, axis=-2)

    # calculate x, y offsets
    grid_x = tf.range(lw, dtype=tf.float32)
    grid_y = tf.range(lh, dtype=tf.float32)
    x_offset, y_offset = tf.meshgrid(grid_x, grid_y)

    shape = x_offset.shape.as_list()
    assert shape[0] == lh and shape[1] == lw
    shape = y_offset.shape.as_list()
    assert shape[0] == lh and shape[1] == lw

    x_offset = tf.expand_dims(x_offset, axis=0)
    y_offset = tf.expand_dims(y_offset, axis=0)

    result = []

    for idx, p in enumerate(priors):
        loc = loc_split[idx]
        obj = obj_split[idx]
        cls = cls_split[idx]

        x, y, w, h = tf.split(loc, [1, 1, 1, 1], axis=-1)

        # squeeze dim one axis from splitting by prior
        x = tf.squeeze(x, axis=[-2, -1])  # don't squeeze the batch axis
        y = tf.squeeze(y, axis=[-2, -1])  # don't squeeze the batch axis
        w = tf.squeeze(w, axis=[-2, -1])  # don't squeeze the batch axis
        h = tf.squeeze(h, axis=[-2, -1])  # don't squeeze the batch axis

        cls = tf.squeeze(cls, axis=[-2])

        # calc bbox coordinates
        x = (x_offset + tf.sigmoid(x)) / lw
        y = (y_offset + tf.sigmoid(y)) / lh
        w = (tf.exp(w) * p.w)
        h = (tf.exp(h) * p.h)

        # center + width and height -> upper left and lower right corner
        w2 = w / 2
        h2 = h / 2
        x0 = x - w2
        y0 = y - h2
        x1 = x + w2
        y1 = y + h2

        # store everything in one tensor with dim N x (4 + 1 + cls_cnt)
        bbox = tf.stack([y0, x0, y1, x1], axis=-1)
        bbox = tf.concat([bbox, obj, cls], axis=-1)
        result.append(bbox)

    return result


def decode_bbox_aleatoric(det, priors, layer_id):
    """
    Converts center x, center y, width and height values to coordinates of top left and bottom right points.

    :param predictions: outputs of YOLO v3 detector of shape (?, h, w, boxes_per_cell * (detection_size))
    :return: converted detections of same shape as input
    """

    batches, lh, lw, box_cnt, cls_cnt = det['cls'].shape.as_list()
    assert box_cnt == len(priors)

    loc_var_ = tf.exp(det['log_loc_var'])
    obj_ = tf.sigmoid(det['obj'])
    cls_ = tf.nn.softmax(det['cls'])

    obj_entropy_ = logistic_entropy(obj_)
    cls_entropy_ = softmax_entropy(cls_)

    loc_split = tf.split(det['loc'], [1] * box_cnt, axis=-2)
    loc_var_split = tf.split(loc_var_, [1] * box_cnt, axis=-2)

    obj_split = tf.split(obj_, [1] * box_cnt, axis=-1)
    obj_entropy_split = tf.split(obj_entropy_, [1] * box_cnt, axis=-1)

    cls_split = tf.split(cls_, [1] * box_cnt, axis=-2)
    cls_entropy_split = tf.split(cls_entropy_, [1] * box_cnt, axis=-1)

    # calculate x, y offsets
    grid_x = tf.range(lw, dtype=tf.float32)
    grid_y = tf.range(lh, dtype=tf.float32)
    x_offset, y_offset = tf.meshgrid(grid_x, grid_y)

    shape = x_offset.shape.as_list()
    assert shape[0] == lh and shape[1] == lw
    shape = y_offset.shape.as_list()
    assert shape[0] == lh and shape[1] == lw

    x_offset = tf.expand_dims(x_offset, axis=0)
    y_offset = tf.expand_dims(y_offset, axis=0)

    result = []

    for idx, p in enumerate(priors):
        loc = loc_split[idx]
        loc_var = loc_var_split[idx]
        obj = obj_split[idx]
        obj_entropy = obj_entropy_split[idx]
        cls = cls_split[idx]
        cls_entropy = cls_entropy_split[idx]

        x, y, w, h = tf.split(loc, [1, 1, 1, 1], axis=-1)

        # squeeze dim one axis from splitting by prior
        x = tf.squeeze(x, axis=[-2, -1])  # don't squeeze the batch axis
        y = tf.squeeze(y, axis=[-2, -1])  # don't squeeze the batch axis
        w = tf.squeeze(w, axis=[-2, -1])  # don't squeeze the batch axis
        h = tf.squeeze(h, axis=[-2, -1])  # don't squeeze the batch axis

        loc_var = tf.squeeze(loc_var, axis=[-2])
        cls = tf.squeeze(cls, axis=[-2])

        # calc bbox coordinates
        x = (x_offset + tf.sigmoid(x)) / lw
        y = (y_offset + tf.sigmoid(y)) / lh
        w = (tf.exp(w) * p.w)
        h = (tf.exp(h) * p.h)

        # center + width and height -> upper left and lower right corner
        w2 = w / 2
        h2 = h / 2
        x0 = x - w2
        y0 = y - h2
        x1 = x + w2
        y1 = y + h2

        loc_ale_total_var = tf.reduce_prod(loc_var, axis=-1)

        ones = tf.ones_like(cls_entropy)

        # store everything in one tensor with dim N x ((4 + 4 + 1) + (1 + 1) + (cls_cnt + 1))
        bbox = tf.stack([y0, x0, y1, x1], axis=-1)
        bbox = tf.concat([bbox, loc_var, tf.expand_dims(loc_ale_total_var, axis=-1), obj, obj_entropy, cls, cls_entropy,
                          layer_id * ones, idx * ones], axis=-1)  # layer_id, prior_id], axis=-1)
        result.append(bbox)

    return result


def logistic_entropy(scores):
    no_obj = (1 - scores) * tf.log((1 - scores))
    obj = scores * tf.log(scores)
    entropy = -(no_obj + obj)  # Note: there is a minus sign!
    return entropy


def softmax_entropy(scores):
    entropy = - tf.reduce_sum(scores * tf.log(scores), axis=-1)  # Note: there is a minus sign!
    return entropy


def decode_epistemic(det):
    """
    Calc mean, var and classification uncertainty for T forward passes of the same image
    """

    loc = det['loc']
    loc_var = tf.exp(det['log_loc_var'])
    obj = tf.sigmoid(det['obj'])
    obj_stddev = tf.exp(det['log_obj_stddev'])  # ignore
    cls = tf.nn.softmax(det['cls'])
    cls_stddev = tf.exp(det['log_cls_stddev'])  # ignore

    # localization (co)variance
    loc_col = tf.expand_dims(loc, axis=-1)  # last two dimensions represent a column vector
    loc_row = tf.expand_dims(loc, axis=-2)  # last two dimenstion represent a row vector

    ev_loc = tf.reduce_mean(loc, axis=0)
    ev_loc_col = tf.expand_dims(ev_loc, axis=-1)  # last two dimensions represent a column vector
    ev_loc_row = tf.expand_dims(ev_loc, axis=-2)  # last two dimenstion represent a row vector

    ev_loc_locT = tf.reduce_mean(loc_col * loc_row, axis=0)  # E[loc_col * loc_row]  (4 x 4)

    epi_covar_loc = ev_loc_locT - (ev_loc_col * ev_loc_row)
    ale_var_loc = tf.reduce_mean(loc_var, axis=0)

    # class and objectness uncertainty
    obj_mean = tf.reduce_mean(obj, axis=0)
    obj_predictive_entropy = logistic_entropy(obj_mean)
    obj_posterior_entropy = tf.reduce_mean(logistic_entropy(obj), axis=0)
    obj_mutual_info = obj_predictive_entropy - obj_posterior_entropy

    cls_mean = tf.reduce_mean(cls, axis=0)
    cls_predictive_entropy = softmax_entropy(cls_mean)
    cls_posterior_entropy = tf.reduce_mean(softmax_entropy(cls), axis=0)
    cls_mutual_info = cls_predictive_entropy - cls_posterior_entropy

    return {
        'ev_loc': ev_loc,  # shape=(lh, lw, box_cnt, 4)
        'epi_covar_loc': epi_covar_loc,  # shape=(lh, lw, box_cnt, 4, 4)
        'ale_var_loc': ale_var_loc,  # shape=(lh, lw, box_cnt, 4)

        'obj_samples': obj,  # shape=(T, lh, lw, box_cnt)  # TODO currently irrelevant
        'obj_mean': obj_mean,  # shape=(lh, lw, box_cnt)
        'obj_mutual_info': obj_mutual_info,  # shape=(lh, lw, box_cnt)
        'obj_entropy': obj_predictive_entropy,  # shape=(lh, lw, box_cnt)

        'cls_samples': cls,  # shape=(T, lh, lw, box_cnt, cls_cnt)  # TODO currently irrelevant
        'cls_mean': cls_mean,  # shape=(lh, lw, box_cnt, cls_cnt)
        'cls_mutual_info': cls_mutual_info,  # shape=(lh, lw, box_cnt)
        'cls_entropy': cls_predictive_entropy,  # shape=(lh, lw, box_cnt)
    }


def decode_bbox_epistemic(det_epistemic, priors, layer_id):
    T, lh, lw, box_cnt, cls_cnt = det_epistemic[
        'cls_samples'].shape.as_list()  # T == number of forward passes for same image
    assert box_cnt == len(priors)

    ev_loc_split = tf.split(det_epistemic['ev_loc'], [1] * box_cnt, axis=-2)
    epi_covar_loc_split = tf.split(det_epistemic['epi_covar_loc'], [1] * box_cnt, axis=-3)
    ale_var_loc_split = tf.split(det_epistemic['ale_var_loc'], [1] * box_cnt, axis=-2)

    obj_mean_split = tf.split(det_epistemic['obj_mean'], [1] * box_cnt, axis=-1)
    obj_mutual_info_split = tf.split(det_epistemic['obj_mutual_info'], [1] * box_cnt, axis=-1)
    obj_entropy_split = tf.split(det_epistemic['obj_entropy'], [1] * box_cnt, axis=-1)

    cls_mean_split = tf.split(det_epistemic['cls_mean'], [1] * box_cnt, axis=-2)
    cls_mutual_info_split = tf.split(det_epistemic['cls_mutual_info'], [1] * box_cnt, axis=-1)
    cls_entropy_split = tf.split(det_epistemic['cls_entropy'], [1] * box_cnt, axis=-1)

    result = []

    # calculate x, y offsets
    grid_x = tf.range(lw, dtype=tf.float32)
    grid_y = tf.range(lh, dtype=tf.float32)
    x_offset, y_offset = tf.meshgrid(grid_x, grid_y)

    ones = tf.ones(shape=(lh, lw, 1), dtype=tf.float32)

    shape = x_offset.shape.as_list()
    assert shape[0] == lh and shape[1] == lw
    shape = y_offset.shape.as_list()
    assert shape[0] == lh and shape[1] == lw

    for idx, p in enumerate(priors):
        ev_loc = ev_loc_split[idx]

        epi_covar_loc = epi_covar_loc_split[idx]
        ale_var_loc = ale_var_loc_split[idx]

        obj_mean = obj_mean_split[idx]
        obj_mutual_info = obj_mutual_info_split[idx]
        obj_entropy = obj_entropy_split[idx]

        cls_mean = cls_mean_split[idx]
        cls_mutual_info = cls_mutual_info_split[idx]
        cls_entropy = cls_entropy_split[idx]

        x, y, w, h = tf.split(ev_loc, [1, 1, 1, 1], axis=-1)

        # squeeze dim one axis from splitting
        x = tf.squeeze(x, axis=[-2, -1])
        y = tf.squeeze(y, axis=[-2, -1])
        w = tf.squeeze(w, axis=[-2, -1])
        h = tf.squeeze(h, axis=[-2, -1])

        epi_covar_loc = tf.squeeze(epi_covar_loc, axis=[-3])
        ale_var_loc = tf.squeeze(ale_var_loc, axis=[-2])
        cls_mean = tf.squeeze(cls_mean, axis=[-2])

        # calc bbox coordinates
        x = (x_offset + tf.sigmoid(x)) / lw
        y = (y_offset + tf.sigmoid(y)) / lh
        w = (tf.exp(w) * p.w)
        h = (tf.exp(h) * p.h)

        # center + width and height -> upper left and lower right corner
        w2 = w / 2
        h2 = h / 2
        x0 = x - w2
        y0 = y - h2
        x1 = x + w2
        y1 = y + h2
        # store everything in one tensor with dim:
        # N x ((4 + 4 + 4 + 1 + 1) + (1 + 1) + (cls_cnt + 1 + 1))
        #      (localization)      + (obj)   + (cls)

        loc_epi_total_var = tf.linalg.det(epi_covar_loc)
        loc_ale_var = tf.reduce_sum(ale_var_loc, axis=-1)
        bbox = tf.stack([y0, x0, y1, x1], axis=-1)

        epi_loc_var = tf.linalg.diag_part(epi_covar_loc)
        bbox = tf.concat([bbox,
                          epi_loc_var, ale_var_loc,  # epistemic and aleatoric var of x, y, w, h
                          tf.expand_dims(loc_epi_total_var, axis=-1),  # total var epi
                          tf.expand_dims(loc_ale_var, axis=-1),  # total var ale
                          obj_mean, obj_mutual_info, obj_entropy,
                          cls_mean, cls_mutual_info, cls_entropy,
                          layer_id * ones, idx * ones], axis=-1)  # layer_id, prior_id
        result.append(bbox)

    return result


def residual(inputs, shortcut):
    inputs = inputs + shortcut
    return inputs


def darknet_batch_norm(inputs, training, trainable):
    inputs = tf.layers.batch_normalization(inputs, training=training, trainable=trainable, epsilon=1e-05)
    return inputs


def batch_norm(inputs, training):
    inputs = tf.layers.batch_normalization(inputs, training=training,
                                           epsilon=1e-05)
    return inputs


def dropout(inputs, drop_prob, standard_test_dropout=False):
    training = not standard_test_dropout
    inputs = tf.layers.dropout(inputs, rate=drop_prob, training=training)  # we always want dropout
    return inputs


def darknet_conv(inputs, filters, kernel_size, strides, training, trainable, weight_regularizer):
    assert kernel_size in [1, 3], 'invalid kernel size'
    assert strides in [1, 2], 'invalid strides'
    if not trainable:
        assert not training

    if strides > 1:
        # the padding in tensorflow and darknet framework differ (darknet is the same as cafe)
        # https://stackoverflow.com/questions/42924324/tensorflows-asymmetric-padding-assumptions
        inputs = darknet_downsample_padding(inputs, kernel_size)
        padding = 'VALID'
    else:
        padding = 'SAME'

    normalizer = {'type': 'darknet_bn', 'training': training}
    return conv(inputs, filters, kernel_size, strides, normalizer, trainable, weight_regularizer, padding=padding)


def conv(inputs, filters, kernel_size, strides, normalizer, trainable, weight_regularizer, padding='SAME'):
    assert kernel_size in [1, 3], 'invalid kernel size'
    assert strides in [1, 2], 'invalid strides'

    use_bias = False
    inputs = tf.layers.conv2d(inputs, filters, kernel_size, strides=strides, activation=None, padding=padding,
                              use_bias=use_bias,
                              trainable=trainable,
                              kernel_regularizer=weight_regularizer,
                              bias_regularizer=weight_regularizer if use_bias else None)

    # check for multiple normalizers:
    if isinstance(normalizer, dict):
        normalizer = [normalizer]

    # possible to add multiple normalizers (first dropout then batch norm)
    for n in normalizer:
        if n['type'] == 'bn':
            inputs = batch_norm(inputs, training=n['training'])
        elif n['type'] == 'darknet_bn':
            inputs = darknet_batch_norm(inputs, training=n['training'], trainable=trainable)
        elif n['type'] == 'dropout':
            if n.get('standard_test_dropout', False):
                dropout(inputs, n['drop_prob'], standard_test_dropout=True)
            else:
                inputs = dropout(inputs, n['drop_prob'])
        elif n['type'] is not None:
            raise ValueError('Invalid regularizer type: {}'.format(n['type']))

    inputs = tf.nn.leaky_relu(inputs, alpha=0.1)
    return inputs


def upsample(inputs):
    shape = tf.shape(inputs)  # NHWC
    return tf.image.resize_nearest_neighbor(inputs, (2 * shape[1], 2 * shape[2]))  # upsample by factor 2


def route(routes):
    assert len(routes) < 3, 'too many routes'
    assert len(routes), 'too few routes'

    if len(routes) > 1:
        inputs = tf.concat(routes, axis=3)  # concatenate channels (if channels_first, then axis=1)
    else:
        inputs = tf.identity(routes[0])  # use identity layer to avoid name clashing when loading darknet weights

    return inputs


def stack_feature_map(inputs, T):
    inputs = tf.concat([inputs] * T, axis=0)
    return inputs


def detection(inputs, cls_cnt, box_cnt, weight_regularizer):
    filters = box_cnt * (4 + 1 + cls_cnt)
    # use linear activation!
    inputs = tf.layers.conv2d(inputs, filters, 1, strides=1, activation=None, padding='SAME',
                              kernel_regularizer=weight_regularizer, bias_regularizer=weight_regularizer)
    return inputs


def detection_aleatoric(inputs, cls_cnt, box_cnt, weight_regularizer):
    filters = box_cnt * (2 * (4 + 1 + cls_cnt))
    # use linear activation!
    inputs = tf.layers.conv2d(inputs, filters, 1, strides=1, activation=None, padding='SAME',
                              kernel_regularizer=weight_regularizer, bias_regularizer=weight_regularizer)
    return inputs


def darknet_downsample_padding(inputs, kernel_size):
    """
    the padding in tensorflow and darknet framework differ, darknet is the same as cafe, see:
    https://stackoverflow.com/questions/42924324/tensorflows-asymmetric-padding-assumptions

    For us this only makes a difference when downsampling (conv2d 3x3 filter and stride 2).

    Note: Maxpooling with a 2x2 kernels also differs between tensorflow and the darknet framework.
          However since we do not use maxpool it is ignored here.
    :param inputs:
    :param kernel_size:
    :return:
    """
    assert kernel_size == 3, 'invalid kernel size'

    pad_front = 1  # this differs from the standard tf padding when: stride = 2, kernel_size = 3 and input_size is even
    pad_end = 1  # could be 0 if input_size is odd, but the overhead of padding is negligible

    inputs = tf.pad(inputs, [[0, 0], [pad_front, pad_end], [pad_front, pad_end], [0, 0]], mode='CONSTANT')
    return inputs
