import json
import logging
import os

from lib_yolo import yolov3, train, utils


def main():
    config = {
        'training': True,  # edit: set to False for qualitative evaluation
        'resume_training': True,
        'resume_checkpoint': './checkpoints/pretraining/pretraining-125000',  # edit
        # 'resume_checkpoint': 'last',  # edit: either filename or 'last' to resume a training
        'priors': yolov3.ECP_9_PRIORS,  # edit if not ECP dataset
        'checkpoint_path': './checkpoints',
        'tensorboard_path': './tensorboard',
        'log_path': './log',
        'ckp_max_to_keep': 75,
        'checkpoint_interval': 5000,
        'ign_thresh': 0.7,
        'crop_img_size': [768, 1440, 3],
        'full_img_size': [1024, 1920, 3],  # edit if not ECP dataset
        'train_steps': 500000,
        'darknet53_weights': './darknet53.conv.74',
        'batch_size': 2,  # edit
        'lr': 1e-5,
        'run_id': 'epi_ale',
        'cpu_thread_cnt': 24,  # edit
        'crop': True,  # edit, random crops and rescaling reduces memory consumption and improves training
        'freeze_darknet53': True,  # if True the basenet weights are frozen during training
        'inference_mode': False,
        'aleatoric_loss': True,
        'cls_cnt': 2,  # edit if not ECP dataset
        'implicit_background_class': True,  # whether the label ids start at 1 or 0. True = 1, False = 0
        'train': {
            'file_pattern': os.path.expandvars('$HOME/data/ecp/tfrecords/ecp-day-train-*-of-*'),  # edit
            'num_shards': 20,
            'shuffle_buffer_size': 2000,
            'cache': False,  # edit if you have enough memory, caches whole dataset in memory
        },
        'val': {
            'file_pattern': os.path.expandvars('$HOME/data/ecp/tfrecords/ecp-day-val-*-of-*'),  # edit
            'num_shards': 4,
            'shuffle_buffer_size': 10,
            'cache': False,  # edit if you have enough memory, caches whole dataset in memory
        }
    }

    # Note regarding implicit background class:
    # The tensorflow object detection API enforces that the class labels start with 1.
    # The class 0 is reserved for an (implicit) background class. We support both file formats.

    utils.add_file_logging(config, override_existing=True)
    logging.info(json.dumps(config, indent=4, default=lambda x: str(x)))

    model_cls = yolov3.bayesian_yolov3_aleatoric

    if config['training']:
        train.start(model_cls, config)
    else:
        config['inference_mode'] = True
        config['resume_checkpoint'] = 'last'
        config['thresh'] = 0.1  # filter out boxes with objectness score less than thresh
        config['T'] = 20  # increase if you have enough memory
        utils.qualitative_eval(model_cls, config)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s, %(levelname)-8s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        )
    main()
