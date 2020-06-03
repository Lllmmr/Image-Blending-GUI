from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from datetime import datetime
import time
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import math
import os
from six.moves import cPickle as pickle
from six.moves import range

from skimage import data, img_as_float
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_mse as mse
from skimage.measure import compare_psnr as psnr

import deepimageharm.deploy_seg as deploy_seg

def _parse_image(composit,image_size:int=256):
    im_ori = composit.copy()
    #im_ori = im_ori.convert('RGB')
    im = cv2.resize(im_ori,(image_size,image_size),cv2.INTER_CUBIC)
    im = np.array(im, dtype=np.float32)
    if im.shape[2] == 4:
        im = im[:, :, 0:3]
    im = im[:, :, ::-1]
    im -= np.array((127.5, 127.5, 127.5))
    im = np.divide(im, np.array(127.5))
    image = im[np.newaxis, ...]
    return image


def _parse_mask(mask,image_size:int =256):
    #mask = Image.open(mask_path)
    mask = cv2.resize(mask,(image_size,image_size),cv2.INTER_CUBIC)
    #mask = mask.resize(np.array([image_size, image_size]), Image.BICUBIC)
    mask = np.array(mask, dtype=np.float32)
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    mask -= 127.5
    mask = np.divide(mask, np.array(127.5))
    mask = mask[np.newaxis, ...]
    mask = mask[..., np.newaxis]
    return mask


def _parse_truth_eval(truth_path,image_size):
    truth = Image.open(truth_path)
    truth = truth.convert('RGB')
    truth = truth.resize(np.array([image_size, image_size]), Image.BICUBIC)
    truth = np.array(truth, dtype=np.float32)
    if truth.ndim == 2:
        truth = truth.reshape(np.array((512, 512, 3)))
    return truth

def MaskAndMove(src,dst,mask,pos):

    shape = dst.shape
    newSrc = np.zeros_like(dst)
    newmask = np.zeros(shape)
    if len(mask.shape)==3 :
        m = mask[:,:,0]
    else:
        m = mask
    x,y,w,h =cv2.boundingRect(m)
    print(h)

    h, w = h // 2, w // 2

    left = min(x, pos[0] - w)
    right = min(src.shape[1] - x - w * 2, shape[1] - pos[0] - w)
    top = min(y, pos[1] - h)
    bottom = min(src.shape[0] - y - h * 2, shape[0] - pos[1] - h)

    newSrc[pos[1] - h - top:pos[1] + h + bottom, pos[0] - w - left:pos[0] + w + right] = src[y - top:y + h * 2 + bottom,
                                                                                         x - left:x + w * 2 + right]
    # cv2.imshow()

    newmask[pos[1] - h:pos[1] + h, pos[0] - w:pos[0] + w] = mask[y:y + h*2, x:x + w*2]

    return newSrc,newmask


def DIH(src,dst,mask,pos):
    FLAGS = tf.app.flags.FLAGS

    image_size = 256

    newsrc , newmask = MaskAndMove(src,dst,mask,pos)
    m = newmask.copy()
    m[m != 0] =1
    composite = m*newsrc +(1-m)*dst

    com_placeholder = tf.placeholder(tf.float32,
                                     shape=(1, image_size, image_size, 3))
    masks_placeholder = tf.placeholder(tf.float32,
                                       shape=(1, image_size, image_size, 1))

    pred_label, harmnization = deploy_seg.inference(com_placeholder, masks_placeholder)

    saver = tf.train.Saver()

    #gpu_options = tf.GPUOptions(allow_growth=True)
    #sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess = tf.Session()
    sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))
    new_saver = tf.train.import_meta_graph('deepimageharm/model/model.ckpt-689457.meta')
    new_saver.restore(sess, 'deepimageharm/model/model.ckpt-689457')
    if new_saver:
        print("Restore successfully!")
    com = _parse_image(composite,image_size)
    mask = _parse_mask(newmask,image_size)

    feed_dict = {
        com_placeholder: com,
        masks_placeholder: mask,
    }
    harm = sess.run(harmnization, feed_dict=feed_dict)
    harm_rgb = np.squeeze(harm)
    harm_rgb = np.multiply(harm_rgb, np.array(127.5))
    harm_rgb += np.array((127.5, 127.5, 127.5))
    harm_rgb = harm_rgb[:, :, ::-1]
    neg_idx = harm_rgb < 0.0
    harm_rgb[neg_idx] = 0.0
    pos_idx = harm_rgb > 255.0
    harm_rgb[pos_idx] = 255.0

    blur = harm_rgb.astype(np.uint8)


    w, h, _ = composite.shape
    #blur = cv2.cvtColor(blur, cv2.COLOR_RGB2BGR)
    blur = cv2.resize(blur, (h, w))

    composite = composite.astype(np.uint8)

    result = cv2.ximgproc.guidedFilter(composite, blur, 20, 2)
    #result = blur

    sess.close()

    return result

if __name__ == '__main__':
    src = cv2.imread("./source1.jpg")
    dst = cv2.imread("./target1.jpg")
    mask = cv2.imread("./mask1.png")

    result = DIH(src, dst, mask, (300, 200))

    cv2.imwrite("rr.jpg", result)