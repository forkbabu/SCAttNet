# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 10:26:49 2018

@author: lhf
"""

import tensorflow as tf
from data import next_batch
from fcn8s import VGG16
import os
import matplotlib.pyplot as plt
import numpy as np
batch_size=16
img=tf.compat.v1.placeholder(tf.float32,[batch_size,256,256,3])
label=tf.compat.v1.placeholder(tf.int32,[batch_size,256,256])
pred=VGG16(img)
cross_entropy_loss = tf.reduce_mean(input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=pred))
global_step=tf.Variable(0,trainable=False)
lr=tf.compat.v1.train.exponential_decay(0.001,global_step=global_step,decay_steps=1000,decay_rate=0.9)
train_step = tf.compat.v1.train.AdamOptimizer(learning_rate=lr).minimize(cross_entropy_loss)
add_global=global_step.assign_add(1)
num_batches=12000//batch_size
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

    
    for i in range(start_epoch,50):
        for j in range(start_batch_id,12000//batch_size):
                x_batch,y_batch=next_batch()
                feed_dict = {   img: x_batch,
                                label: y_batch
                            }
                _,loss,pred1=sess.run([train_step,cross_entropy_loss,pred],feed_dict=feed_dict)
                _,lr_rate=sess.run([add_global,lr])

                    
                print('epoch',i,'|loss',loss,'|lr',lr_rate)
                counter+=1
        start_batch_id=0
        saver.save(sess,'./checkpoint/fcn32s.ckpt',global_step=counter)

def test():
    
    for j in range(0,len(test_img)//batch_size):
                
                x_batch=test_img[batch_size*j:batch_size*(j+1)]
                y_batch=test_labels[batch_size*j:batch_size*(j+1)]
                feed_dict = {   img: x_batch,
                                label: y_batch
                            }
                loss,pred1=sess.run([cross_entropy_loss,pred],feed_dict=feed_dict)
                pred_map_batch = np.argmax(pred1, axis=3)
                plt.subplot(131)
                plt.imshow(pred_map_batch[0])
                plt.subplot(132)
                plt.imshow(y_batch[0])
                plt.subplot(133)
                plt.imshow(x_batch[0])
                print('loss',loss)
                
    
    
with tf.compat.v1.Session() as sess:
    train()
    
