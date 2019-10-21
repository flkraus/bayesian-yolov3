import tensorflow as tf

from lib_yolo import darknet, model, data


def _get_city_persons_9_priors():
    priors = [[495.27, 203.83],
              [297.84, 122.19],
              [197.44, 81.48],
              [141.07, 58.5],
              [102.72, 43.1],
              [75.78, 31.66],
              [54.24, 23.19],
              [37.55, 16.15],
              [22.55, 10.09]]

    priors = [[p[0] / 1024., p[1] / 2048.] for p in priors]  # priors are calculated for original citypersons img size
    priors_32 = [data.Prior(h=p[0], w=p[1]) for p in priors[:3]]
    priors_16 = [data.Prior(h=p[0], w=p[1]) for p in priors[3:6]]
    priors_8 = [data.Prior(h=p[0], w=p[1]) for p in priors[6:]]

    return {
        32: priors_32,
        16: priors_16,
        8: priors_8,
    }


def _get_ecp_9_priors():
    priors = [
        [0.56643243, 0.13731691],
        [0.41022839, 0.09028599],
        [0.30508716, 0.06047965],
        [0.20774711, 0.04376083],
        [0.15475611, 0.02996197],
        [0.10878717, 0.02149197],
        [0.07694039, 0.01488527],
        [0.05248527, 0.01007212],
        [0.03272104, 0.00631827],
    ]

    # # in pixels:
    # priors = [[580.02680832, 263.64846719999997],
    #           [420.07387136, 173.3491008],
    #           [312.40925184, 116.120928],
    #           [212.73304064, 84.0207936],
    #           [158.47025664, 57.5269824],
    #           [111.39806208, 41.264582399999995],
    #           [78.78695936, 28.5797184],
    #           [53.74491648, 19.338470400000002],
    #           [33.50634496, 12.1310784]]

    priors_32 = [data.Prior(h=p[0], w=p[1]) for p in priors[:3]]
    priors_16 = [data.Prior(h=p[0], w=p[1]) for p in priors[3:6]]
    priors_8 = [data.Prior(h=p[0], w=p[1]) for p in priors[6:]]

    return {
        32: priors_32,
        16: priors_16,
        8: priors_8,
    }


def _get_ecp_night_9_priors():
    priors = [
        [0.6197282176953125, 0.14694562146874998],
        [0.4243941425683594, 0.09687759120833334],
        [0.3103862368359375, 0.06362734035416667],
        [0.23494613041992188, 0.043568554453125],
        [0.1634832566796875, 0.03293052755208333],
        [0.12444031231445313, 0.023274527578125],
        [0.08800429220703125, 0.016930080526041665],
        [0.06101826478515625, 0.011638404229166668],
        [0.03925641140625, 0.007475639645833334],
    ]

    # # in pixels:
    # priors = [[634.60169492, 282.13559322],
    #           [434.57960199, 186.00497512],
    #           [317.83550652, 122.16449348],
    #           [240.58483755, 83.65162455],
    #           [167.40685484, 63.2266129],
    #           [127.42687981, 44.68709295],
    #           [90.11639522, 32.50575461],
    #           [62.48270314, 22.34573612],
    #           [40.19856528, 14.35322812]]

    priors_32 = [data.Prior(h=p[0], w=p[1]) for p in priors[:3]]
    priors_16 = [data.Prior(h=p[0], w=p[1]) for p in priors[3:6]]
    priors_8 = [data.Prior(h=p[0], w=p[1]) for p in priors[6:]]

    return {
        32: priors_32,
        16: priors_16,
        8: priors_8,
    }


def _get_ecp_day_night_9_priors():
    priors = [
        [0.5728529907421875, 0.13943622409895834],
        [0.41761617583007815, 0.09156660707291667],
        [0.3015263176855469, 0.06248444700520834],
        [0.22101856140625, 0.042888710765625],
        [0.1533158565527344, 0.031196821406250002],
        [0.11255495265625, 0.021566710822916668],
        [0.07823327209960937, 0.015212825187500001],
        [0.0533416983203125, 0.010216603067708333],
        [0.0332035418359375, 0.006413999807291667]
    ]

    # # in pixels:
    # priors = [[586.60146252, 267.71755027],
    #           [427.63896405, 175.80788558],
    #           [308.76294931, 119.97013825],
    #           [226.32300688, 82.34632467],
    #           [156.99543711, 59.8978971],
    #           [115.25627152, 41.40808478],
    #           [80.11087063, 29.20862436],
    #           [54.62189908, 19.61587789],
    #           [34.00042684, 12.31487963]]

    priors_32 = [data.Prior(h=p[0], w=p[1]) for p in priors[:3]]
    priors_16 = [data.Prior(h=p[0], w=p[1]) for p in priors[3:6]]
    priors_8 = [data.Prior(h=p[0], w=p[1]) for p in priors[6:]]

    return {
        32: priors_32,
        16: priors_16,
        8: priors_8,
    }


def _get_ecp_with_bic_9_priors():
    priors = [
        [0.5541169062011718, 0.15767184942708334],
        [0.3872792363671875, 0.08849276056770834],
        [0.27297898112304686, 0.05552458755208333],
        [0.18570756796875, 0.034849724458333335],
        [0.13080457012695312, 0.052510955223958336],
        [0.12203939466796875, 0.02422101765625],
        [0.083340965234375, 0.01635016602083333],
        [0.055563667021484374, 0.010672233619791667],
        [0.03409191838867188, 0.006481136984375],
    ]

    # # in pixels:
    # priors = [[567.41571195, 302.7299509],
    #           [396.57393804, 169.90610029],
    #           [279.53047667, 106.6072081],
    #           [190.1645496, 66.91147096],
    #           [133.94387981, 100.82103403],
    #           [124.96834014, 46.5043539],
    #           [85.3411484, 31.39231876],
    #           [56.89719503, 20.49068855],
    #           [34.91012443, 12.44378301]]

    priors_32 = [data.Prior(h=p[0], w=p[1]) for p in priors[:3]]
    priors_16 = [data.Prior(h=p[0], w=p[1]) for p in priors[3:6]]
    priors_8 = [data.Prior(h=p[0], w=p[1]) for p in priors[6:]]

    return {
        32: priors_32,
        16: priors_16,
        8: priors_8,
    }


CITY_PERSONS_9_PRIORS = _get_city_persons_9_priors()
ECP_9_PRIORS = _get_ecp_9_priors()
ECP_NIGHT_9_PRIORS = _get_ecp_night_9_priors()
ECP_DAY_NIGHT_9_PRIORS = _get_ecp_day_night_9_priors()
ECP_BIC_9_PRIORS = _get_ecp_with_bic_9_priors()


class yolov3:
    def __init__(self, config):
        self.__model = None
        self.img_size, self.__priors = model.img_size_and_priors_if_crop(config)
        self.__darknet53_layer_cnt = 0
        self.__freeze_darknet53 = config.get('freeze_darknet53', True)
        self.cls_cnt = config['cls_cnt']
        self.obj_idx = 4
        self.cls_start_idx = 5

        self.blueprint = model.ModelBlueprint(det_layers=[
            model.DetLayerBlueprint(
                input_img_size=self.img_size,
                downsample_factor=32,
                priors=self.__priors[32],
            ), model.DetLayerBlueprint(
                input_img_size=self.img_size,
                downsample_factor=16,
                priors=self.__priors[16],
            ),
            model.DetLayerBlueprint(
                input_img_size=self.img_size,
                downsample_factor=8,
                priors=self.__priors[8],
            )
        ], cls_cnt=self.cls_cnt)

        # check that input_img_size is multiple of biggest stride
        # otherwise the following happens:
        # ValueError: Dimensions must be equal, but are 16 and 15 for \
        #  'det_net_1/detection/loss/localization/sub' (op: 'Sub') with input shapes: [?,16,32,3,4], [?,15,31,3,4].
        assert config['full_img_size'][0] % 32 == 0
        assert config['full_img_size'][1] % 32 == 0
        if config['crop']:
            assert config['crop_img_size'][0] % 32 == 0
            assert config['crop_img_size'][1] % 32 == 0

    def get_model(self):
        """
        call init_model first!
        """
        assert self.__model is not None, 'Call init_model first.'
        return self.__model

    def load_darknet53_weights(self, weightfile):
        assert self.__model is not None, 'Call init_model first.'
        return darknet.load_darknet_weights(self.__model.layers[:self.__darknet53_layer_cnt], weightfile)

    def init_model(self, inputs, training, gt1=None, gt2=None, gt3=None):
        if self.__model is not None:
            raise Exception('model can only be initialized once!')

        self.__build_model(inputs, training, gt1, gt2, gt3)
        assert self.__model.matches_blueprint(self.blueprint), 'Model does not match blueprint'
        return self

    def __build_model(self, inputs, training, gt1=None, gt2=None, gt3=None):
        in_shape = inputs.get_shape().as_list()
        assert len(in_shape) == 4, 'invalid data format'
        # assert in_shape[3] == 3, 'invalid data format'

        mb = model.ModelBuilder(inputs=inputs, cls_cnt=self.cls_cnt)
        normalizer = {'type': 'bn', 'training': training}

        with tf.variable_scope('darknet53'):
            darknet53_training = False if self.__freeze_darknet53 else training
            darknet53_trainable = not self.__freeze_darknet53
            darknet.darknet53(mb, training=darknet53_training, trainable=darknet53_trainable)  # 0 - 74

        dn_out = mb.inputs
        self.__darknet53_layer_cnt = mb.layer_cnt()

        with tf.variable_scope('det_net_1'):
            mb.make_conv_layer(512, 1, normalizer)  # .................... # 75
            mb.make_conv_layer(1024, 3, normalizer)  # ................... # 76

            mb.make_conv_layer(512, 1, normalizer)  # .................... # 77
            mb.make_conv_layer(1024, 3, normalizer)  # ................... # 78

            mb.make_conv_layer(512, 1, normalizer)  # .................... # 79
            mb.make_conv_layer(1024, 3, normalizer)  # ................... # 80

            mb.make_detection_layer(all_priors=self.__priors, gt=gt1)  # .. # 81
            # YOLO LAYER  # ............................................... # 82
            det_net_1_out = mb.inputs

        with tf.variable_scope('det_net_2'):
            # -3 instead of -4, since we don't add a YOLO layer to layers list.
            mb.make_route_layer([-3])  # .................................. # 83
            mb.make_conv_layer(256, 1, normalizer)  # .................... # 84

            # Downsample (factor 16)
            mb.make_upsample_layer()  # ................................... # 85
            mb.make_route_layer([-1, 61])  # .............................. # 86

            mb.make_conv_layer(256, 1, normalizer)  # .................... # 87
            mb.make_conv_layer(512, 3, normalizer)  # .................... # 88

            mb.make_conv_layer(256, 1, normalizer)  # .................... # 89
            mb.make_conv_layer(512, 3, normalizer)  # .................... # 90

            mb.make_conv_layer(256, 1, normalizer)  # .................... # 91
            mb.make_conv_layer(512, 3, normalizer)  # .................... # 92

            mb.make_detection_layer(all_priors=self.__priors, gt=gt2)  # .. # 93
            # YOLO LAYER # ................................................ # 94
            det_net_2_out = mb.inputs

        with tf.variable_scope('det_net_3'):
            # -3 instead of -4, since we don't add a YOLO layer to layers list.
            mb.make_route_layer([-3])  # .................................. # 95
            mb.make_conv_layer(128, 1, normalizer)  # .................... # 96

            # Downsample (factor 8)
            mb.make_upsample_layer()  # ................................... # 97
            mb.make_route_layer([-1, 36])  # .............................. # 98

            mb.make_conv_layer(128, 1, normalizer)  # .................... # 99
            mb.make_conv_layer(256, 3, normalizer)  # .................... # 100

            mb.make_conv_layer(128, 1, normalizer)  # .................... # 101
            mb.make_conv_layer(256, 3, normalizer)  # .................... # 102

            mb.make_conv_layer(128, 1, normalizer)  # .................... # 103
            mb.make_conv_layer(256, 3, normalizer)  # .................... # 104

            mb.make_detection_layer(all_priors=self.__priors, gt=gt3)  # .. # 105
            # YOLO LAYER  # ............................................... # 106
            det_net_3_out = mb.inputs

        self.__model = mb.get_model(self.obj_idx, self.cls_start_idx)
        self.__model.dn_out = dn_out
        self.__model.det_net_1_out = det_net_1_out
        self.__model.det_net_2_out = det_net_2_out
        self.__model.det_net_3_out = det_net_3_out


class yolov3_aleatoric:
    def __init__(self, config):
        self.__aleatoric_loss = config['aleatoric_loss']
        self.__model = None
        self.img_size, self.__priors = model.img_size_and_priors_if_crop(config)
        self.__darknet53_layer_cnt = 0
        self.__freeze_darknet53 = config.get('freeze_darknet53', True)
        self.cls_cnt = config['cls_cnt']
        self.obj_idx = 9
        self.cls_start_idx = 11

        self.blueprint = model.ModelBlueprint(det_layers=[
            model.DetLayerBlueprint(
                input_img_size=self.img_size,
                downsample_factor=32,
                priors=self.__priors[32],
            ), model.DetLayerBlueprint(
                input_img_size=self.img_size,
                downsample_factor=16,
                priors=self.__priors[16],
            ),
            model.DetLayerBlueprint(
                input_img_size=self.img_size,
                downsample_factor=8,
                priors=self.__priors[8],
            )
        ], cls_cnt=self.cls_cnt)

        # check that input_img_size is multiple of biggest stride
        # otherwise the following happens:
        # ValueError: Dimensions must be equal, but are 16 and 15 for \
        #  'det_net_1/detection/loss/localization/sub' (op: 'Sub') with input shapes: [?,16,32,3,4], [?,15,31,3,4].
        assert config['full_img_size'][0] % 32 == 0
        assert config['full_img_size'][1] % 32 == 0
        if config['crop']:
            assert config['crop_img_size'][0] % 32 == 0
            assert config['crop_img_size'][1] % 32 == 0

    def get_model(self):
        """
        call init_model first!
        """
        assert self.__model is not None, 'Call init_model first.'
        return self.__model

    def load_darknet53_weights(self, weightfile):
        assert self.__model is not None, 'Call init_model first.'
        return darknet.load_darknet_weights(self.__model.layers[:self.__darknet53_layer_cnt], weightfile)

    def init_model(self, inputs, training, gt1=None, gt2=None, gt3=None):
        if self.__model is not None:
            raise Exception('model can only be initialized once!')

        self.__build_model(inputs, training, gt1, gt2, gt3)
        assert self.__model.matches_blueprint(self.blueprint), 'Model does not match blueprint'
        return self

    def __build_model(self, inputs, training, gt1=None, gt2=None, gt3=None):
        in_shape = inputs.get_shape().as_list()
        assert len(in_shape) == 4, 'invalid data format'
        # assert in_shape[3] == 3, 'invalid data format'

        mb = model.ModelBuilder(inputs=inputs, cls_cnt=self.cls_cnt)
        normalizer = {'type': 'bn', 'training': training}

        with tf.variable_scope('darknet53'):
            darknet53_training = False if self.__freeze_darknet53 else training
            darknet53_trainable = not self.__freeze_darknet53
            darknet.darknet53(mb, training=darknet53_training, trainable=darknet53_trainable)  # 0 - 74

        dn_out = mb.inputs
        self.__darknet53_layer_cnt = mb.layer_cnt()

        with tf.variable_scope('det_net_1'):
            mb.make_conv_layer(512, 1, normalizer)  # .................... # 75
            mb.make_conv_layer(1024, 3, normalizer)  # ................... # 76

            mb.make_conv_layer(512, 1, normalizer)  # .................... # 77
            mb.make_conv_layer(1024, 3, normalizer)  # ................... # 78

            mb.make_conv_layer(512, 1, normalizer)  # .................... # 79
            mb.make_conv_layer(1024, 3, normalizer)  # ................... # 80

            mb.make_detection_layer_aleatoric(all_priors=self.__priors, aleatoric_loss=self.__aleatoric_loss,
                                              gt=gt1)  # .. # 81
            # YOLO LAYER  # ............................................... # 82
            det_net_1_out = mb.inputs

        with tf.variable_scope('det_net_2'):
            # -3 instead of -4, since we don't add a YOLO layer to layers list.
            mb.make_route_layer([-3])  # .................................. # 83
            mb.make_conv_layer(256, 1, normalizer)  # .................... # 84

            # Downsample (factor 16)
            mb.make_upsample_layer()  # ................................... # 85
            mb.make_route_layer([-1, 61])  # .............................. # 86

            mb.make_conv_layer(256, 1, normalizer)  # .................... # 87
            mb.make_conv_layer(512, 3, normalizer)  # .................... # 88

            mb.make_conv_layer(256, 1, normalizer)  # .................... # 89
            mb.make_conv_layer(512, 3, normalizer)  # .................... # 90

            mb.make_conv_layer(256, 1, normalizer)  # .................... # 91
            mb.make_conv_layer(512, 3, normalizer)  # .................... # 92

            mb.make_detection_layer_aleatoric(all_priors=self.__priors, aleatoric_loss=self.__aleatoric_loss,
                                              gt=gt2)  # .. # 93
            # YOLO LAYER # ................................................ # 94
            det_net_2_out = mb.inputs

        with tf.variable_scope('det_net_3'):
            # -3 instead of -4, since we don't add a YOLO layer to layers list.
            mb.make_route_layer([-3])  # .................................. # 95
            mb.make_conv_layer(128, 1, normalizer)  # .................... # 96

            # Downsample (factor 8)
            mb.make_upsample_layer()  # ................................... # 97
            mb.make_route_layer([-1, 36])  # .............................. # 98

            mb.make_conv_layer(128, 1, normalizer)  # .................... # 99
            mb.make_conv_layer(256, 3, normalizer)  # .................... # 100

            mb.make_conv_layer(128, 1, normalizer)  # .................... # 101
            mb.make_conv_layer(256, 3, normalizer)  # .................... # 102

            mb.make_conv_layer(128, 1, normalizer)  # .................... # 103
            mb.make_conv_layer(256, 3, normalizer)  # .................... # 104

            mb.make_detection_layer_aleatoric(all_priors=self.__priors, aleatoric_loss=self.__aleatoric_loss,
                                              gt=gt3)  # .. # 105
            # YOLO LAYER  # ............................................... # 106
            det_net_3_out = mb.inputs

        self.__model = mb.get_model(self.obj_idx, self.cls_start_idx)
        self.__model.dn_out = dn_out
        self.__model.det_net_1_out = det_net_1_out
        self.__model.det_net_2_out = det_net_2_out
        self.__model.det_net_3_out = det_net_3_out


class bayesian_yolov3_aleatoric:
    def __init__(self, config):
        self.__aleatoric_loss = config['aleatoric_loss']
        self.__model = None
        self.img_size, self.__priors = model.img_size_and_priors_if_crop(config)
        self.__darknet53_layer_cnt = 0
        self.__freeze_darknet53 = config.get('freeze_darknet53', True)
        self.__inference_mode = config['inference_mode']
        self.__drop_prob = 0.1
        self.cls_cnt = config['cls_cnt']
        self.obj_idx = 14
        self.cls_start_idx = 17

        if self.__inference_mode:
            self.__T = config['T']

        self.__standard_test_dropout = config.get('standard_test_dropout', False)

        self.blueprint = model.ModelBlueprint(det_layers=[
            model.DetLayerBlueprint(
                input_img_size=self.img_size,
                downsample_factor=32,
                priors=self.__priors[32],
            ), model.DetLayerBlueprint(
                input_img_size=self.img_size,
                downsample_factor=16,
                priors=self.__priors[16],
            ),
            model.DetLayerBlueprint(
                input_img_size=self.img_size,
                downsample_factor=8,
                priors=self.__priors[8],
            )
        ], cls_cnt=self.cls_cnt)

        # check that input_img_size is multiple of biggest stride
        # otherwise the following happens:
        # ValueError: Dimensions must be equal, but are 16 and 15 for \
        #  'det_net_1/detection/loss/localization/sub' (op: 'Sub') with input shapes: [?,16,32,3,4], [?,15,31,3,4].
        assert config['full_img_size'][0] % 32 == 0
        assert config['full_img_size'][1] % 32 == 0
        if config['crop']:
            assert config['crop_img_size'][0] % 32 == 0
            assert config['crop_img_size'][1] % 32 == 0

    def get_model(self):
        """
        call init_model first!
        """
        assert self.__model is not None, 'Call init_model first.'
        return self.__model

    def load_darknet53_weights(self, weightfile):
        assert self.__model is not None, 'Call init_model first.'
        return darknet.load_darknet_weights(self.__model.layers[:self.__darknet53_layer_cnt], weightfile)

    def init_model(self, inputs, training, gt1=None, gt2=None, gt3=None):
        if self.__model is not None:
            raise Exception('model can only be initialized once!')

        self.__build_model(inputs, training, gt1, gt2, gt3)
        assert self.__model.matches_blueprint(self.blueprint), 'Model does not match blueprint'
        return self

    def __build_model(self, inputs, training, gt1=None, gt2=None, gt3=None):
        in_shape = inputs.get_shape().as_list()
        assert len(in_shape) == 4, 'invalid data format'
        # assert in_shape[3] == 3, 'invalid data format'

        mb = model.ModelBuilder(inputs=inputs, cls_cnt=self.cls_cnt)
        bn = {'type': 'bn', 'training': training}
        dropout_bn = [
            {'type': 'dropout', 'drop_prob': self.__drop_prob, 'standard_test_dropout': self.__standard_test_dropout},
            bn
        ]  # add batch norm after dropout

        with tf.variable_scope('darknet53'):
            darknet53_training = False if self.__freeze_darknet53 else training
            darknet53_trainable = not self.__freeze_darknet53
            darknet.darknet53(mb, training=darknet53_training, trainable=darknet53_trainable)  # 0 - 74

        dn_out = mb.inputs
        self.__darknet53_layer_cnt = mb.layer_cnt()

        if self.__inference_mode:
            # stack dn_out N times
            # make route layer to stacked dn => this messes with the layer numbering => should be fine in inference mode
            mb.make_stack_feature_map_layer(-1, self.__T)  # additional layer, now the counting is of (shouldn't matter)

        with tf.variable_scope('det_net_1'):
            mb.make_conv_layer(512, 1, dropout_bn)  # ..................... # 75
            mb.make_conv_layer(1024, 3, dropout_bn)  # .................... # 76

            mb.make_conv_layer(512, 1, dropout_bn)  # ..................... # 77
            mb.make_conv_layer(1024, 3, dropout_bn)  # .................... # 78

            mb.make_conv_layer(512, 1, dropout_bn)  # ..................... # 79
            mb.make_conv_layer(1024, 3, bn)  # ............................ # 80

            mb.make_detection_layer_aleatoric_epistemic(
                all_priors=self.__priors,
                aleatoric_loss=self.__aleatoric_loss,
                gt=gt1,
                inference_mode=self.__inference_mode,
            )  # .......................................................... # 81
            # YOLO LAYER  # ............................................... # 82
            det_net_1_out = mb.inputs

        with tf.variable_scope('det_net_2'):
            # -3 instead of -4, since we don't add a YOLO layer to layers list.
            mb.make_route_layer([-3])  # .................................. # 83
            mb.make_conv_layer(256, 1, bn)  # ............................. # 84

            # Downsample (factor 16)
            mb.make_upsample_layer()  # ................................... # 85
            if self.__inference_mode:
                mb.make_stack_feature_map_layer(61, self.__T)
                mb.make_route_layer([-2, -1])  # .......................... # 86
            else:
                mb.make_route_layer([-1, 61])  # .......................... # 86

            mb.make_conv_layer(256, 1, dropout_bn)  # ..................... # 87
            mb.make_conv_layer(512, 3, dropout_bn)  # ..................... # 88

            mb.make_conv_layer(256, 1, dropout_bn)  # ..................... # 89
            mb.make_conv_layer(512, 3, dropout_bn)  # ..................... # 90

            mb.make_conv_layer(256, 1, dropout_bn)  # ..................... # 91
            mb.make_conv_layer(512, 3, bn)  # ............................. # 92

            mb.make_detection_layer_aleatoric_epistemic(
                all_priors=self.__priors,
                aleatoric_loss=self.__aleatoric_loss,
                gt=gt2,
                inference_mode=self.__inference_mode,
            )  # .......................................................... # 93
            # YOLO LAYER # ................................................ # 94
            det_net_2_out = mb.inputs

        with tf.variable_scope('det_net_3'):
            # -3 instead of -4, since we don't add a YOLO layer to layers list.
            mb.make_route_layer([-3])  # .................................. # 95
            mb.make_conv_layer(128, 1, bn)  # ............................. # 96

            # Downsample (factor 8)
            mb.make_upsample_layer()  # ................................... # 97
            if self.__inference_mode:
                mb.make_stack_feature_map_layer(36, self.__T)
                mb.make_route_layer([-2, -1])  # .......................... # 98
            else:
                mb.make_route_layer([-1, 36])  # .......................... # 98

            mb.make_conv_layer(128, 1, dropout_bn)  # ..................... # 99
            mb.make_conv_layer(256, 3, dropout_bn)  # ..................... # 100

            mb.make_conv_layer(128, 1, dropout_bn)  # ..................... # 101
            mb.make_conv_layer(256, 3, dropout_bn)  # ..................... # 102

            mb.make_conv_layer(128, 1, dropout_bn)  # ..................... # 103
            mb.make_conv_layer(256, 3, bn)  # ............................. # 104

            mb.make_detection_layer_aleatoric_epistemic(
                all_priors=self.__priors,
                aleatoric_loss=self.__aleatoric_loss,
                gt=gt3,
                inference_mode=self.__inference_mode,
            )  # .......................................................... # 105
            # YOLO LAYER  # ............................................... # 106
            det_net_3_out = mb.inputs

        self.__model = mb.get_model(self.obj_idx, self.cls_start_idx)
        self.__model.dn_out = dn_out
        self.__model.det_net_1_out = det_net_1_out
        self.__model.det_net_2_out = det_net_2_out
        self.__model.det_net_3_out = det_net_3_out
