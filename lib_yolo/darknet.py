import os

import numpy as np
import tensorflow as tf


def darknet53(model_builder, training, trainable):
    mb = model_builder

    mb.make_darknet_conv_layer(32, 3, training, trainable)  # 0

    # Downsample (factor 2)
    mb.make_darknet_downsample_layer(64, 3, training, trainable)  # 1

    mb.make_darknet_residual_block(32, training, trainable)  # 2 - 4

    # Downsample (factor 4)
    mb.make_darknet_downsample_layer(128, 3, training, trainable)  # 5

    for i in range(2):
        mb.make_darknet_residual_block(64, training, trainable)  # 6 - 11

    # Downsample (factor 8)
    mb.make_darknet_downsample_layer(256, 3, training, trainable)  # 12

    for i in range(8):
        mb.make_darknet_residual_block(128, training, trainable)  # 13 - 36

    # Downsample (factor 16)
    mb.make_darknet_downsample_layer(512, 3, training, trainable)  # 37

    for i in range(8):
        mb.make_darknet_residual_block(256, training, trainable)  # 38 - 61

    # Downsample (factor 32)
    mb.make_darknet_downsample_layer(1024, 3, training, trainable)  # 62

    for i in range(4):
        mb.make_darknet_residual_block(512, training, trainable)  # 63 - 74


def load_darknet_weights(net_layers, weightfile):
    with open(weightfile, "rb") as f:
        header = np.fromfile(f, dtype=np.int32, count=5)
        weights = np.fromfile(f, dtype=np.float32)

    ptr = 0
    tmp = tf.global_variables()
    vars = {}
    for var in tmp:
        vars[var.name] = var

    assign_ops = []

    for i, l in enumerate(net_layers):
        if 'LeakyRelu' not in l.name:
            continue

        batch_norm = 'detection' not in l.name
        load_bias = not batch_norm
        if batch_norm:
            ptr = _load_batch_norm(l, vars, ptr, weights, assign_ops)

        ptr = _load_conv2d(l, vars, ptr, weights, assign_ops, load_bias)

    assert ptr == len(weights)
    return assign_ops


def _load_conv2d(l, vars, ptr, weights, assign_ops, load_bias):
    namespace = l.name.split('/')
    namespace = os.path.join(*namespace[:2], 'conv2d')

    kernel_name = os.path.join(namespace, 'kernel:0')
    kernel = vars[kernel_name]

    if load_bias:
        bias_name = os.path.join(namespace, 'bias:0')
        bias = vars[bias_name]

        bias_shape = bias.shape.as_list()
        bias_params = np.prod(bias_shape)
        bias_weights = weights[ptr:ptr + bias_params].reshape(bias_shape)
        ptr += bias_params
        assign_ops.append(tf.assign(bias, bias_weights, validate_shape=True))

    kernel_shape = kernel.shape.as_list()
    kernel_params = np.prod(kernel_shape)

    [h, w, c, n] = kernel_shape
    kernel_weights = weights[ptr:ptr + kernel_params].reshape([n, c, h, w])
    # transpose to [h, w, c, n]
    kernel_weights = np.transpose(kernel_weights, (2, 3, 1, 0))

    ptr += kernel_params
    assign_ops.append(tf.assign(kernel, kernel_weights, validate_shape=True))

    return ptr


def _load_batch_norm(l, vars, ptr, weights, assign_ops):
    namespace = l.name.split('/')
    namespace = os.path.join(*namespace[:2], 'batch_normalization')

    gamma = os.path.join(namespace, 'gamma:0')
    beta = os.path.join(namespace, 'beta:0')
    moving_mean = os.path.join(namespace, 'moving_mean:0')
    moving_variance = os.path.join(namespace, 'moving_variance:0')

    gamma = vars[gamma]
    beta = vars[beta]
    moving_mean = vars[moving_mean]
    moving_variance = vars[moving_variance]

    for var in [beta, gamma, moving_mean, moving_variance]:
        shape = var.shape.as_list()
        num_params = np.prod(shape)
        var_weights = weights[ptr:ptr + num_params].reshape(shape)
        ptr += num_params
        assign_ops.append(tf.assign(var, var_weights, validate_shape=True))

    return ptr
