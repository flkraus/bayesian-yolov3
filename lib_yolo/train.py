import datetime
import json
import logging
import os

import numpy as np
import tensorflow as tf

from lib_yolo import dataset_utils, data_augmentation


def save_config(config, folder):
    time_string = datetime.datetime.now().isoformat().split('.')[0]
    file = os.path.join(folder, 'config_{}_{}.json'.format(time_string, config['run_id']))

    try:
        os.makedirs(folder)
    except IOError:
        pass

    with open(file, 'w') as f:
        json.dump(config, f, indent=4, default=lambda x: str(x))


def start(model_cls, config):
    if config['crop']:
        cropper = data_augmentation.ImageCropper(config)
        config['train']['crop_fn'] = cropper.random_crop_and_sometimes_rescale
        config['val']['crop_fn'] = cropper.random_crop_and_sometimes_rescale

    model_factory = model_cls(config)

    dataset = dataset_utils.TrainValDataset(model_blueprint=model_factory.blueprint, config=config)

    # currently all models have 3 detection layers, so this works...
    img, gt1, gt2, gt3 = dataset.iterator.get_next()
    model = model_factory.init_model(inputs=img, training=True, gt1=gt1, gt2=gt2, gt3=gt3).get_model()

    # also all models are powerd by darknet53...
    assign_ops = model_factory.load_darknet53_weights(config['darknet53_weights'])

    with tf.Session(config=tf.ConfigProto(device_count={'GPU': 1})) as sess:
        dataset.init_dataset(sess)
        try:
            train(sess, model, dataset, config, init_ops=assign_ops)
        except:
            logging.exception('ERROR')
            raise


def train(sess, model, dataset, config, init_ops=None):
    def train_loop_body():
        summary, tloss, dloss, rloss, lloss, oloss, closs = sess.run(
            [train_step, summary_op, model.total_loss, model.detection_loss, model.regularization_loss,
             model.loc_loss, model.obj_loss, model.cls_loss], feed_dict={dataset.handle: dataset.train_handle})[1:]
        if np.isnan(tloss) or np.isinf(tloss):
            logging.error('{:5d} >>> total_loss: {:8.2f}, det_loss: {:8.2f}, loc_loss: {:8.2f},'
                          ' obj_loss: {:8.2f}, cls_loss: {:8.2f}, reg_loss: {:8.5f}'.format(
                step, tloss, dloss, lloss, oloss, closs, rloss))
            return False

        if step % 25 == 0:
            writer_train.add_summary(summary, step)
            logging.info('{:5d} train >>> total_loss: {:8.2f}, det_loss: {:8.2f}, loc_loss: {:8.2f},'
                         ' obj_loss: {:8.2f}, cls_loss: {:8.2f}, reg_loss: {:8.5f}'.format(
                step, tloss, dloss, lloss, oloss, closs, rloss))

        if step % 100 == 0:
            summary, tloss, dloss, rloss, lloss, oloss, closs = sess.run(
                [summary_op, model.total_loss, model.detection_loss, model.regularization_loss,
                 model.loc_loss, model.obj_loss, model.cls_loss],
                feed_dict={dataset.handle: dataset.val_handle})
            writer_val.add_summary(summary, step)
            logging.info(
                '{:5d} val   >>> total_loss: {:8.2f}, det_loss: {:8.2f}, loc_loss: {:8.2f},'
                ' obj_loss: {:8.2f}, cls_loss: {:8.2f}, reg_loss: {:8.5f}'.format(
                    step, tloss, dloss, lloss, oloss, closs, rloss))

        if step % config['checkpoint_interval'] == 0:
            saver.save(sess, os.path.join(save_path, config['run_id']), global_step=step)

        return True

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.variable_scope('optimizer'):
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate=config['lr'])
            train_step = optimizer.minimize(model.total_loss, var_list=tf.trainable_variables())

    # TODO create hist summaries
    summary_op = tf.summary.merge_all()

    saver = tf.train.Saver(max_to_keep=config['ckp_max_to_keep'])
    save_path = os.path.join(config['checkpoint_path'], config['run_id'])
    save_config(config, save_path)

    if config['resume_training']:
        checkpoint = config['resume_checkpoint']
        if checkpoint == 'last':
            checkpoint = tf.train.latest_checkpoint(save_path)
        step = int(checkpoint.split('-')[-1])
        saver.restore(sess, checkpoint)
    else:
        # It is important to first run the global variable initializer and then the init_ops!
        # Otherwise the initializations of init_ops would be overridden.
        sess.run(tf.global_variables_initializer())
        if init_ops:
            sess.run(init_ops)
        step = 0

    writer_train = tf.summary.FileWriter(os.path.join(config['tensorboard_path'], config['run_id'], 'train'),
                                         sess.graph)
    writer_val = tf.summary.FileWriter(os.path.join(config['tensorboard_path'], config['run_id'], 'val'))
    try:
        while step < config['train_steps']:
            step += 1
            success = train_loop_body()
            if not success:
                logging.error('An error occurred, abort training.')
                break
    except KeyboardInterrupt:
        # gracefully exit
        logging.info('KeyboardInterrupt: Abort training.')
        ans = ''
        while ans.lower() not in ['yes', 'y', 'no', 'n']:
            ans = input('Save checkpoint (yes/no): ')
        if ans.lower() in ['no', 'n']:
            return  # without saving checkpoint
    except:
        # try to save if an unexpected error occurs
        logging.error('Unexpected error occured, try to save checkpoint.')
        saver.save(sess, os.path.join(save_path, config['run_id']), global_step=step)

    # save if training ended
    saver.save(sess, os.path.join(save_path, config['run_id']), global_step=step)
