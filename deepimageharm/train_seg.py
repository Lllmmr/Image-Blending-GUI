from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time
import numpy as np
from PIL import Image
import tensorflow as tf
import os

import deploy_seg

tf.reset_default_graph()

FLAGS = tf.app.flags.FLAGS
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())

tf.app.flags.DEFINE_integer('epoch', 1000,
                           """Number of epochs to run trainer.""")

TRAIN_DIR = 'logs/'+TIMESTAMP+'_e'+str(FLAGS.epoch)+'_lr'+str(FLAGS.init_lr)+'_b'+str(FLAGS.batch_size)+'/'

IMAGE_SIZE = 256
TRAIN_AMOUNT = 38545
STEP_PER_TRAIN_EPOCH = int(TRAIN_AMOUNT / FLAGS.batch_size) 

def train():
  image_path, mask_path, truth_path, label_path = deploy_seg.inputs()
  
  with tf.Graph().as_default():
    global_step = tf.train.get_or_create_global_step()

    image_batch, mask_batch, truth_batch, label_batch = deploy_seg.get_batch_pre(image_path, mask_path, truth_path, label_path)

    images_placeholder = tf.placeholder(tf.float32,
                                    shape=(FLAGS.batch_size, IMAGE_SIZE, IMAGE_SIZE, 3),name='image_plh')
    masks_placeholder = tf.placeholder(tf.float32,
                                    shape=(FLAGS.batch_size, IMAGE_SIZE, IMAGE_SIZE, 1),name='mask_plh')
    truths_placeholder = tf.placeholder(tf.float32,
                                    shape=(FLAGS.batch_size, IMAGE_SIZE, IMAGE_SIZE, 3),name='truth_plh')
    label_placeholder = tf.placeholder(tf.float32,
                                    shape=(FLAGS.batch_size, IMAGE_SIZE, IMAGE_SIZE),name='label_plh')

    pred_label, composite = deploy_seg.inference(images_placeholder,masks_placeholder)

    parse_com = deploy_seg._parse_eval_(composite)
    tf.summary.image('parsed_composite', parse_com, max_outputs=20)
    parse_label = deploy_seg._parse_label_(pred_label)
    tf.summary.image('parsed_label', parse_label, max_outputs=20)

    loss = deploy_seg.loss_pre(pred_label, composite, label_placeholder, truths_placeholder)
    train_op = deploy_seg.train(loss,global_step)
    init = tf.global_variables_initializer()
    summary = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep=7)
    checkpoint_file = os.path.join(TRAIN_DIR, 'model.ckpt')
    with tf.Session() as sess:
        sess.run(init)
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord = coord)
        writer_train = tf.summary.FileWriter(TRAIN_DIR, sess.graph)
        sess.graph.finalize()
        print("Initialized")
        start_time = time.time()
        for e in range(FLAGS.epoch):
          for step in range(STEP_PER_TRAIN_EPOCH):
            image_batch_v, mask_batch_v, truth_batch_v,label_batch_v = sess.run([image_batch, mask_batch, truth_batch, label_batch])
            feed_dict_t={
                    images_placeholder:image_batch_v,
                    masks_placeholder:mask_batch_v,
                    truths_placeholder:truth_batch_v,
                    label_placeholder:label_batch_v
                    }
            _, loss_value, train_pre = sess.run(
                [[pred_label, composite], loss, train_op], feed_dict=feed_dict_t)
            if((step+1) % 200 ==0):
              duration = time.time() - start_time
              print('Epoch %d step %d: total loss = %.7f (%.3f sec)' %(e, step+1, loss_value, duration))
              summary_str = sess.run(summary,feed_dict=feed_dict_t)
              writer_train.add_summary(summary_str,step)
              writer_train.flush()
          if((e+1) %3 == 0):
              saver.save(sess, checkpoint_file, global_step=global_step) 
        coord.request_stop()
        coord.join(threads)



def main(argv=None):
  if tf.gfile.Exists(TRAIN_DIR):
    tf.gfile.DeleteRecursively(TRAIN_DIR)
  tf.gfile.MakeDirs(TRAIN_DIR)
  train()

if __name__ == '__main__':
  tf.app.run()