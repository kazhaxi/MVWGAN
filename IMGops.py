"""
For any questions, please email me at
If using this code, parts of it, or developments from it, please cite our paper:

Thank you and good luck!
"""

# -*- coding: utf-8 -*-
from __future__ import division
import os
import scipy.misc
import numpy as np
from glob import glob


# Loading pearl data and labels
def center_crop(img, crop_h, crop_w):
    if crop_w is None:
        crop_w = crop_h
    h,w = img.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return img[j:j+crop_h, i:i+crop_w]


def load_one_pearl(start_number, crop_h, crop_w):
    data_list1 = glob("./data/pic0/*.jpg")
    data_list2 = glob("./data/pic1/*.jpg")
    data_list3 = glob("./data/pic2/*.jpg")
    data_list4 = glob("./data/pic3/*.jpg")
    data_list5 = glob("./data/pic4/*.jpg")

    five_img = scipy.misc.imread(data_list1[start_number]).astype(np.float)
    img = scipy.misc.imread(data_list2[start_number]).astype(np.float)
    five_img = np.concatenate((five_img, img), axis=2)
    img = scipy.misc.imread(data_list3[start_number]).astype(np.float)
    five_img = np.concatenate((five_img, img), axis=2)
    img = scipy.misc.imread(data_list4[start_number]).astype(np.float)
    five_img = np.concatenate((five_img, img), axis=2)
    img = scipy.misc.imread(data_list5[start_number]).astype(np.float)
    five_img = np.concatenate((five_img, img), axis=2)
    #center crop
    five_img = center_crop(five_img, crop_h, crop_w)
    # Normalization
    return five_img/255


def load_one_label(y_dim,data_list):
    label = data_list.split('\\')[-1]
    label = label.split('_')[0]
    label = int(label)
    label_vec = np.zeros((y_dim), dtype=np.float)
    label_vec[label] = 1.0
    return label_vec


def save_images(images, save_dir, epoch, idx):
    for j in range(images.shape[0]):
        img = images[j]
        for i in range(0, 5):
            save_path = os.path.join(save_dir,
            'tr_ep{}_b{}_{}_{}.jpg'.format(epoch, idx, j, i))
            img1 = img[:, :, i*3:(i+1)*3]
            img1 = np.uint8(img1 * 255)
            scipy.misc.imsave(save_path, img1)


def test_save_images(images, save_dir, class_num, loop):
    for j in range(images.shape[0]):
        img = images[j]
        for i in range(0, 5):
            img_zeros = np.zeros((300, 300, 3))
            save_dir_a = os.path.join(save_dir,'pic{}'.format(i))
            save_dir_i = os.path.join(save_dir_a, 'train_{}'.format(class_num))
            if not os.path.exists(save_dir_i):
                os.makedirs(save_dir_i)
            save_path = os.path.join(save_dir_i,
            '{}_{}.jpg'.format(class_num, 31+loop+j))
            img1 = img[:, :, i*3:(i+1)*3]
            img_zeros[25:275, 25:275, :]=img1
            img_zeros = np.uint8(img_zeros * 255)
            scipy.misc.imsave(save_path, img_zeros)


def visualize(sess, pearl_gan, config):
    for i in range(27):
        test_z = np.random.uniform(-0.99, 0.99, size=(config.batch_size, pearl_gan.z_dim))
        test_sample = sess.run(pearl_gan.sampler,
                               	feed_dict={
                                      	 pearl_gan.z:test_z
                                       	})
        test_save_images(test_sample, config.sample_dir, config.test_class, config.batch_size*i)
              

def load_one_pearl_generate(start_number, crop_h, crop_w):
    data_list1 = glob("./samples/pic0/train_1/*.jpg")
    data_list2 = glob("./samples/pic1/train_1/*.jpg")
    data_list3 = glob("./samples/pic2/train_1/*.jpg")
    data_list4 = glob("./samples/pic3/train_1/*.jpg")
    data_list5 = glob("./samples/pic4/train_1/*.jpg")

    five_img = scipy.misc.imread(data_list1[start_number]).astype(np.float)
    img = scipy.misc.imread(data_list2[start_number]).astype(np.float)
    five_img = np.concatenate((five_img, img), axis=2)
    img = scipy.misc.imread(data_list3[start_number]).astype(np.float)
    five_img = np.concatenate((five_img, img), axis=2)
    img = scipy.misc.imread(data_list4[start_number]).astype(np.float)
    five_img = np.concatenate((five_img, img), axis=2)
    img = scipy.misc.imread(data_list5[start_number]).astype(np.float)
    five_img = np.concatenate((five_img, img), axis=2)
    # Center crop
    five_img = center_crop(five_img, crop_h, crop_w)
    # Normalization
    return five_img/255


def load_one_label_number(data_list):
    label = data_list.split('\\')[-1]
    label = label.split('_')[0]
    label = int(label)
    return label