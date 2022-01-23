"""
For any questions, please email me at
If using this code, parts of it, or developments from it, please cite our paper:

Thank you and good luck!
"""

# -*- coding: utf-8 -*-
import os
import numpy as np
from pearl_model import pearl_GAN
import pprint
from IMGops import *
import tensorflow as tf
import tensorflow.contrib.slim as slim

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

flags = tf.app.flags
flags.DEFINE_integer("epoch", 1400, "Epoch to train [1400]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "`Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", 20000000, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 10, "The size of batch images [10]")
flags.DEFINE_integer("input_height", 250, "The size of image to use (will be center cropped). [250]")
flags.DEFINE_integer("input_width", 250, "The size of image to use (will be center cropped). If None, same value as input_height [250]")
flags.DEFINE_string("input_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", "./checkpoints", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "./sample_30", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
flags.DEFINE_integer("test_class", 0, "The option of test [0,1,2,3,4,5,6]")
FLAGS = flags.FLAGS

def main(_):
    pp = pprint.PrettyPrinter()
    pp.pprint(flags.FLAGS.__flags)

    if FLAGS.input_width is None:
        FLAGS.input_width = FLAGS.input_height

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    # GPU
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth=True

    with tf.Session(config=run_config) as sess:
        pearl_gan = pearl_GAN(
                sess,
                batch_size = FLAGS.batch_size,
                crop_height = FLAGS.input_height,
                crop_width = FLAGS.input_width,
                input_pattern = FLAGS.input_pattern,
                checkpoint_dir = FLAGS.checkpoint_dir,
                sample_dir = FLAGS.sample_dir
                )

        show_all_variables()

        if FLAGS.train:
            pearl_gan.train(FLAGS)
        else:
            if not pearl_gan.load(FLAGS.checkpoint_dir)[0]:
                raise Exception("[!] Train a model first, then run test mode")

        visualize(sess, pearl_gan, FLAGS)

if __name__ == '__main__':
    tf.app.run()



