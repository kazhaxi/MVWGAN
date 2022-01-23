"""
For any questions, please email me at
If using this code, parts of it, or developments from it, please cite our paper:

Thank you and good luck!
"""

# -*- coding: utf-8 -*-

from __future__ import division
import os
import math
from glob import glob
import tensorflow as tf
from IMGops import *
import numpy as np
from NNops import *
import time


def get_randomlist(num):
    random_list = [i for i in range(num)]
    np.random.seed(123)
    np.random.shuffle(random_list)
    return random_list


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


class pearl_GAN(object):
    def __init__(self, sess, batch_size=16, crop_height=250, crop_width=250,  
                 z_dim=100, c_dim=15, gf_dim=64, df_dim=64,
                 dataset_name="pearl_52500", input_pattern="*.jpg", 
                 checkpoint_dir=None, sample_dir=None
                 ):
        self.sess = sess
        
        self.batch_size = batch_size
        self.input_h = crop_height
        self.input_w = crop_width
        self.c_dim = c_dim
        self.z_dim = z_dim
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        
        #batch_norm
        self.d_bn1 = layer_norm(name='d_bn1')
        self.d_bn2 = layer_norm(name='d_bn2')
        self.d_bn3 = layer_norm(name='d_bn3')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')

        self.dataset_name = dataset_name
        self.input_pattern = input_pattern
        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir

        self.random_list = get_randomlist(30)
        
        self.build_model()
        
    def build_model(self):
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name="noise_z")
        #tensorboard
        self.z_sum = histogram_summary("z", self.z)

        self.inputs = tf.placeholder(
                tf.float32, [self.batch_size, self.input_h, self.input_w, self.c_dim], name="real_image")

        self.G = self.generator(self.z)
        self.D, self.D_logits = self.discriminator(self.inputs, reuse=False)

        self.sampler = self.sampler(self.z)
        self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

        #tensorboard
        self.d_sum = histogram_summary("d", self.D)
        self.d__sum = histogram_summary("d_", self.D_)

        def sigmoid_cross_entropy_with_logits(x, y):
            try:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
            except:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

        self.d_loss = tf.reduce_mean(self.D_logits_) - tf.reduce_mean(self.D_logits)
        self.g_loss = -tf.reduce_mean(self.D_logits_)
        self.Wasserstein_D = -self.d_loss
        # gradient penalty from WGAN-GP
        self.eps = tf.random_uniform([self.batch_size, 1, 1, 1], minval=0., maxval=1.)
        self.x_penalty = self.eps * self.inputs + (1 - self.eps) * self.G
        self.penalty_logits = self.discriminator(self.x_penalty, reuse=True)
        self.gradients = tf.gradients(self.penalty_logits, self.x_penalty)[0]

        # 2-Norm
        self.grad_norm = tf.sqrt(tf.reduce_sum(tf.square(self.gradients), reduction_indices=[1, 2, 3]))
        self.grad_pen = tf.reduce_mean((self.grad_norm - 1.) ** 2)

        self.d_loss = self.d_loss + 10 * self.grad_pen
        # tensorboard
        self.Wasserstein_D_sum = scalar_summary("Wasserstein_D", self.Wasserstein_D)
        self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
        self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

        # Extract the training variables of the two networks separately
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]
        
        self.saver = tf.train.Saver()
    
    def train(self, config):
        d_optim = tf.train.AdamOptimizer(
            config.learning_rate, beta1=config.beta1).minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(
            config.learning_rate, beta1=config.beta1).minimize(self.g_loss, var_list=self.g_vars)
        
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()
        
        self.g_sum = merge_summary(
                [self.z_sum, self.d__sum, self.g_loss_sum])
        self.d_sum = merge_summary(
                [self.z_sum, self.d_sum, self.Wasserstein_D_sum, self.d_loss_sum])
        self.writer = SummaryWriter("./logs", self.sess.graph)
        
        counter = 1
        # Load checkpoint
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load Succes")
        else:
            print(" [*] Load Failed")

        batch_idxs = min(len(self.random_list), config.train_size) // self.batch_size
                        
        # Create sample data to see the effect of training
        sample_z = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))
        sample_imgs = self.load_batch_pearl(
                        self.random_list, 0)
        
        # Iterative process
        start_time = time.time()              
        for epoch in range(config.epoch):
            for idx in range(0, batch_idxs):
                batch_images = self.load_batch_pearl(
                        self.random_list, idx)

                # Sample random noise
                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]) \
                                           .astype(np.float32)
                
                #updata D
                _, summary_str = self.sess.run([d_optim, self.d_sum],
                                               feed_dict={
                                                       self.inputs:batch_images,
                                                       self.z:batch_z
                                                       })
                self.writer.add_summary(summary_str, counter)

                _, summary_str = self.sess.run([d_optim, self.d_sum],
                                               feed_dict={
                                                   self.inputs: batch_images,
                                                   self.z: batch_z
                                               })                # updata D

                self.writer.add_summary(summary_str, counter)
                
                #updata G
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                                               feed_dict={
                                                       self.z:batch_z
                                                       })
                self.writer.add_summary(summary_str, counter)


                errD = self.d_loss.eval({
                        self.inputs:batch_images,self.z:batch_z})
                errG = self.g_loss.eval({
                        self.z:batch_z})
                
                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, idx, batch_idxs,
                         time.time()-start_time, errD, errG))
                
                if np.mod(counter, 300) == 1:
                    samples, d_loss, g_loss = self.sess.run(
                            [self.sampler, self.d_loss, self.g_loss],
                            feed_dict={
                                    self.z:sample_z,
                                    self.inputs:sample_imgs
                                    })
                    save_images(samples, self.sample_dir, epoch, idx)
                    print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

                # Save checkpoint and counter every 300 times
                if np.mod(counter, 300) == 1:
                    self.save(config.checkpoint_dir, counter)

        
    def generator(self, z ):
        with tf.variable_scope("generator") as scope:
            #输出图片大小
            s_h, s_w = self.input_h, self.input_w
            #中间层卷积输出大小，从尾到头
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

            z_ = linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin')
            
            h0 = tf.reshape(z_, [-1, s_h16, s_w16, self.gf_dim*8])
            h0 = tf.nn.relu(self.g_bn0(h0))
            
            h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1')
            h1 = tf.nn.relu(self.g_bn1(h1))

            h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2')
            h2 = tf.nn.relu(self.g_bn2(h2))

            h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3')
            h3 = tf.nn.relu(self.g_bn3(h3))

            h4 = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4')

            return tf.nn.sigmoid(h4)


    def discriminator(self, image, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:                 
                scope.reuse_variables()

            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
            
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
            
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
            h3 = tf.reshape(h3, [self.batch_size, -1])
            
            h4 = linear(h3, 1, 'd_h4_lin')
            
            return tf.nn.sigmoid(h4), h4
        
    def sampler(self, z):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()
            s_h, s_w = self.input_h, self.input_w

            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)
            
            z_ = linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin')
            
            h0 = tf.reshape(z_, [-1, s_h16, s_w16, self.gf_dim*8])
            h0 = tf.nn.relu(self.g_bn0(h0))
            
            h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1')
            h1 = tf.nn.relu(self.g_bn1(h1))
            
            h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2')
            h2 = tf.nn.relu(self.g_bn2(h2))

            h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3')
            h3 = tf.nn.relu(self.g_bn3(h3))
            
            h4 = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4')
            return tf.nn.sigmoid(h4)
    

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
                self.dataset_name, self.batch_size, self.input_h, self.input_w)
    
    def save(self, checkpoint_dir, step):
        model_name = "PearlGAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        self.saver.save(self.sess, 
                        os.path.join(checkpoint_dir, model_name), 
                        global_step=step)
    
    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoint...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Succes to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def load_batch_pearl(self, random_list, idx):
        batch_pearl_list = random_list[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_data = [
                load_one_pearl(i,
                               self.input_h, 
                               self.input_w) for i in batch_pearl_list]
        batch_data = np.array(batch_data).astype(np.float32)

        return batch_data
