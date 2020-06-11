# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 10:26:49 2018

@author: lhf
"""

import tensorflow as tf
from data import next_batch
from scattnet import inference
import os
import matplotlib.pyplot as plt
import numpy as np

batch_size=16
img=tf.compat.v1.placeholder(tf.float32,[batch_size,256,256,3])
label=tf.compat.v1.placeholder(tf.int32,[batch_size,256,256])
phase_train = tf.compat.v1.placeholder(tf.bool, name='phase_train')
pred = inference(img,phase_train)
cross_entropy_loss = tf.reduce_mean(input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=pred))
train_step = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy_loss)
total_number=12000
num_batches=total_number//batch_size   #total number of train image
saver=tf.compat.v1.train.Saver()


def load():
    import re
    print(" [*] Reading checkpoints...")
    checkpoint_dir = './checkpoint/'

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
        print(" [*] Success to read {}".format(ckpt_name))
        return True, counter
    else:
        print(" [*] Failed to find a checkpoint")
        return False, 0

def train():
    tf.compat.v1.global_variables_initializer().run()    
    could_load, checkpoint_counter = load()
    if could_load:
        start_epoch = (int)(checkpoint_counter / num_batches)
        start_batch_id = checkpoint_counter - start_epoch * num_batches
        counter = checkpoint_counter
        print(" [*] Load SUCCESS")
        print(" [*] Load SUCCESS")
    else:
        start_epoch = 0
        start_batch_id = 0
        counter = 1
        print(" [!] Load failed...")

    
    for i in range(start_epoch,100):
        for j in range(start_batch_id,total_number//batch_size):
                x_batch,y_batch=next_batch()
                feed_dict = {   img: x_batch,
                                label: y_batch,
                                phase_train: True   
                            }

                _,loss,pred1=sess.run([train_step,cross_entropy_loss,pred],feed_dict=feed_dict)
                    
                print('epoch',i,'|loss',loss)
                counter+=1
        start_batch_id=0
        saver.save(sess,'./checkpoint/scattnet.ckpt',global_step=counter)

with tf.compat.v1.Session() as sess:
    train()
    
