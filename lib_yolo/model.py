import tensorflow as tf

from lib_yolo import layers, data


def img_size_and_priors_if_crop(config):  # TODO fn name is not very descriptive
    img_size = config['crop_img_size'] if config['crop'] else config['full_img_size']
    priors = config['priors']

    if config['crop']:
        # priors are always defined for the full img => need to rescale if we crop the image
        scale_h = config['full_img_size'][0] / float(config['crop_img_size'][0])
        scale_w = config['full_img_size'][1] / float(config['crop_img_size'][1])
        for stride, prs in priors.items():
            priors[stride] = [data.Prior(h=p.h * scale_h, w=p.w * scale_w) for p in prs]

    return img_size, priors


class ModelBuilder:
    def __init__(self, inputs, cls_cnt, l2_scale=0.0005):
        self.__layers = []
        self.inputs = inputs
        self.__cls_cnt = cls_cnt
        self.__current_downsample = 1
        self.__det_layers = []
        self.__weight_regularizer = tf.contrib.layers.l2_regularizer(l2_scale)

        shape = inputs.shape.as_list()
        assert len(shape) == 4
        self.__input_size = [shape[1], shape[2]]  # h x w

    def layer_cnt(self):
        return len(self.__layers)

    def get_model(self, obj_idx, cls_start_idx):
        assert obj_idx < cls_start_idx
        return Model(self.__layers, self.__det_layers, self.__cls_cnt, obj_idx, cls_start_idx)

    def __update_layers(self):
        self.__layers.append(self.inputs)

    def __conv_layer(self, filters, kernel_size, strides, normalizer, variable_scope):
        assert strides in [1, 2]

        with tf.variable_scope(None, default_name=variable_scope):
            # use_bias = True
            self.inputs = layers.conv(self.inputs, filters, kernel_size, strides, normalizer, trainable=True,
                                      weight_regularizer=self.__weight_regularizer)
            self.__update_layers()

    def make_conv_layer(self, filters, kernel_size, normalizer):
        self.__conv_layer(filters, kernel_size, 1, normalizer, 'conv')

    def make_downsample_layer(self, filters, kernel_size, normalizer):
        self.__current_downsample *= 2
        self.__conv_layer(filters, kernel_size, 2, normalizer, 'downsample')

    def __darknet_conv_layer(self, filters, kernel_size, strides, training, trainable, variable_scope):
        assert strides in [1, 2]
        if not trainable:
            assert not training

        with tf.variable_scope(None, default_name=variable_scope):
            self.inputs = layers.darknet_conv(self.inputs, filters, kernel_size, strides, training, trainable,
                                              self.__weight_regularizer)
            self.__update_layers()

    def make_darknet_conv_layer(self, filters, kernel_size, training, trainable):
        self.__darknet_conv_layer(filters, kernel_size, 1, training, trainable, 'conv')

    def make_darknet_downsample_layer(self, filters, kernel_size, training, trainable):
        self.__current_downsample *= 2
        self.__darknet_conv_layer(filters, kernel_size, 2, training, trainable, 'downsample')

    def make_route_layer(self, routes):
        with tf.variable_scope(None, default_name='route'):
            self.inputs = layers.route([self.__layers[route] for route in routes])
            self.__update_layers()

    def make_stack_feature_map_layer(self, layer, T):
        with tf.variable_scope(None, default_name='stack_feature_map'):
            self.inputs = layers.stack_feature_map(self.__layers[layer], T)
            self.__update_layers()

    def make_residual_layer(self, shortcut):
        with tf.variable_scope(None, default_name='residual'):
            self.inputs = layers.residual(self.inputs, self.__layers[shortcut])
            self.__update_layers()

    def make_residual_block(self, filters, normalizer):
        self.make_conv_layer(filters, 1, normalizer)
        self.make_conv_layer(2 * filters, 3, normalizer)
        self.make_residual_layer(-3)

    def make_darknet_residual_block(self, filters, training, trainable):
        self.make_darknet_conv_layer(filters, 1, training, trainable)
        self.make_darknet_conv_layer(2 * filters, 3, training, trainable)
        self.make_residual_layer(-3)

    def make_upsample_layer(self):
        self.__current_downsample //= 2
        with tf.variable_scope(None, default_name='upsample'):
            self.inputs = layers.upsample(self.inputs)
            self.__update_layers()

    def make_detection_layer(self, all_priors, gt=None):
        priors = all_priors[self.__current_downsample]
        box_cnt = len(priors)
        with tf.variable_scope('detection'):
            self.inputs = layers.detection(self.inputs, cls_cnt=self.__cls_cnt, box_cnt=box_cnt,
                                           weight_regularizer=self.__weight_regularizer)
            self.__update_layers()
            det = layers.split_detection(self.inputs, boxes_per_cell=box_cnt,
                                         cls_cnt=self.__cls_cnt)
            bbox = layers.decode_bbox_standard(det, priors)
            if gt:
                loss = layers.loss_tf(det, gt, aleatoric_loss=False)

        self.__det_layers.append(DetLayer(
            input_img_size=self.__input_size,
            downsample_factor=self.__current_downsample,
            priors=priors,
            loss=loss if gt else None,
            det=det,
            bbox=bbox,
            raw_output=self.inputs,
        ))

        return self.inputs

    def make_detection_layer_aleatoric(self, all_priors, aleatoric_loss, gt=None):
        priors = all_priors[self.__current_downsample]
        box_cnt = len(priors)
        with tf.variable_scope('detection'):
            self.inputs = layers.detection_aleatoric(self.inputs, cls_cnt=self.__cls_cnt, box_cnt=box_cnt,
                                                     weight_regularizer=self.__weight_regularizer)
            self.__update_layers()
            det = layers.split_detection_aleatoric(self.inputs, boxes_per_cell=box_cnt,
                                                   cls_cnt=self.__cls_cnt)
            bbox = layers.decode_bbox_aleatoric(det, priors, layer_id=len(self.__det_layers))
            if gt:
                loss = layers.loss_tf(det, gt, aleatoric_loss)

        self.__det_layers.append(DetLayer(
            input_img_size=self.__input_size,
            downsample_factor=self.__current_downsample,
            priors=priors,
            loss=loss if gt else None,
            det=det,
            bbox=bbox,
            raw_output=self.inputs,
        ))

        return self.inputs

    def make_detection_layer_aleatoric_epistemic(self, all_priors, aleatoric_loss, gt=None, inference_mode=False):
        priors = all_priors[self.__current_downsample]
        box_cnt = len(priors)
        with tf.variable_scope('detection'):
            self.inputs = layers.detection_aleatoric(self.inputs, cls_cnt=self.__cls_cnt, box_cnt=box_cnt,
                                                     weight_regularizer=self.__weight_regularizer)
            self.__update_layers()
            det = layers.split_detection_aleatoric(self.inputs, boxes_per_cell=box_cnt,
                                                   cls_cnt=self.__cls_cnt)
            if inference_mode:
                det = layers.decode_epistemic(det)
                bbox = layers.decode_bbox_epistemic(det, priors, layer_id=len(self.__det_layers))
            else:
                bbox = layers.decode_bbox_aleatoric(det, priors, layer_id=len(self.__det_layers))

            if gt:
                loss = layers.loss_tf(det, gt, aleatoric_loss)

        self.__det_layers.append(DetLayer(
            input_img_size=self.__input_size,
            downsample_factor=self.__current_downsample,
            priors=priors,
            loss=loss if gt else None,
            det=det,
            bbox=bbox,
            raw_output=self.inputs,
        ))

        return self.inputs


class Model:
    def __init__(self, layers, det_layers, cls_cnt, obj_idx, cls_start_idx):
        assert len(det_layers) > 0
        self.layers = layers
        self.det_layers = det_layers
        self.cls_cnt = cls_cnt
        self.obj_idx = obj_idx
        self.cls_start_idx = cls_start_idx

        if self.det_layers[0].loc_loss is not None:
            with tf.name_scope('global_loss'):
                self.detection_loss = tf.losses.get_total_loss(add_regularization_losses=False, name='detection_loss')
                self.regularization_loss = tf.losses.get_regularization_loss(name='regularization_loss')
                self.total_loss = tf.add(self.detection_loss, self.regularization_loss, name='total_loss')

                self.loc_loss = det_layers[0].loc_loss
                self.obj_loss = det_layers[0].obj_loss
                self.cls_loss = det_layers[0].cls_loss
                for l in self.det_layers[1:]:
                    self.loc_loss += l.loc_loss
                    self.obj_loss += l.obj_loss
                    self.cls_loss += l.cls_loss

                tf.summary.scalar('loc', self.loc_loss)
                tf.summary.scalar('obj', self.obj_loss)
                tf.summary.scalar('cls', self.cls_loss)
                tf.summary.scalar('detection', self.detection_loss)
                tf.summary.scalar('l2_weight_reg', self.regularization_loss)
                tf.summary.scalar('total', self.total_loss)

    def matches_blueprint(self, blueprint):
        try:
            for dl, bpdl in zip(self.det_layers, blueprint.det_layers):
                assert dl.matches_blueprint(bpdl)
            assert self.cls_cnt == blueprint.cls_cnt
        except AssertionError:
            return False
        return True


class DetLayer:
    def __init__(self, input_img_size, downsample_factor, priors, loss, det, bbox, raw_output):
        self.h = input_img_size[0] // downsample_factor
        self.w = input_img_size[1] // downsample_factor
        self.downsample = downsample_factor
        self.priors = priors
        self.loc_loss = loss['loc'] if loss else None
        self.obj_loss = loss['obj'] if loss else None
        self.cls_loss = loss['cls'] if loss else None

        self.det = det
        self.bbox = bbox

        self.raw_output = raw_output

    def matches_blueprint(self, blueprint):
        try:
            assert self.h == blueprint.h
            assert self.w == blueprint.w
            assert self.downsample == blueprint.downsample
            assert len(self.priors) == len(blueprint.priors)
            for p, bpp in zip(self.priors, blueprint.priors):
                assert p.h == bpp.h
                assert p.w == bpp.w
        except AssertionError:
            return False
        return True


class ModelBlueprint:
    def __init__(self, det_layers, cls_cnt):
        self.det_layers = det_layers
        self.cls_cnt = cls_cnt


class DetLayerBlueprint:
    def __init__(self, input_img_size, downsample_factor, priors):
        self.h = input_img_size[0] // downsample_factor
        self.w = input_img_size[1] // downsample_factor
        self.downsample = downsample_factor
        self.priors = priors
