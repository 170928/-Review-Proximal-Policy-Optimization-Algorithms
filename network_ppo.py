import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import datetime

action_size = 5
vector_state_size = 6 * 3
visual_state_size = [84, 84, 3]

class network():

    def __init__(self, model_name):

        with tf.variable_scope(model_name):
            self.vector_obs = tf.placeholder(dtype=tf.float32, shape=[None, vector_state_size])
            self.visual_obs = tf.placeholder(dtype=tf.float32, shape=[None, visual_state_size[0], visual_state_size[1], visual_state_size[2]])
            self.normalized_visual_obs = (self.visual_obs - (255.0 / 2)) / (255.0 / 2)

            with tf.variable_scope('policy'):
                layer_1 = tf.layers.dense(inputs=self.vector_obs, units=256, activation=tf.tanh)
                layer_2 = tf.layers.dense(inputs=layer_1, units=256, activation=tf.tanh)
                conv1 = tf.layers.conv2d(inputs=self.normalized_visual_obs, filters=32, activation=tf.nn.relu, kernel_size=[8, 8], strides=[4, 4], padding="SAME")
                conv2 = tf.layers.conv2d(inputs=conv1, filters=64, activation=tf.nn.relu, kernel_size=[4, 4],  strides=[2, 2], padding="SAME")
                conv3 = tf.layers.conv2d(inputs=conv2, filters=64, activation=tf.nn.relu, kernel_size=[3, 3],  strides=[1, 1], padding="SAME")
                flat = tf.layers.flatten(conv3)
                layer_3 = tf.layers.dense(inputs=tf.concat([layer_2, flat], axis = 1), units=128, activation=tf.tanh)
                layer_4 = tf.layers.dense(inputs=layer_3, units=64, activation=tf.tanh)
                layer_5 = tf.layers.dense(inputs=layer_4, units=32, activation=tf.tanh)
                self.act_probs = tf.layers.dense(inputs=layer_5, units=action_size, activation=tf.nn.softmax)

            with tf.variable_scope('value'):
                layer_1 = tf.layers.dense(inputs=self.vector_obs, units=512, activation=tf.nn.tanh)
                layer_2 = tf.layers.dense(inputs=layer_1, units=256, activation=tf.nn.tanh)
                conv1 = tf.layers.conv2d(inputs=self.normalized_visual_obs, filters=32, activation=tf.nn.relu, kernel_size=[8, 8], strides=[4, 4], padding="SAME")
                conv2 = tf.layers.conv2d(inputs=conv1, filters=64, activation=tf.nn.relu, kernel_size=[4, 4],  strides=[2, 2], padding="SAME")
                conv3 = tf.layers.conv2d(inputs=conv2, filters=64, activation=tf.nn.relu, kernel_size=[3, 3],  strides=[1, 1], padding="SAME")
                flat = tf.layers.flatten(conv3)
                layer_3 = tf.layers.dense(inputs=tf.concat([layer_2, flat], axis = 1), units=128, activation=tf.nn.tanh)
                layer_4 = tf.layers.dense(inputs=layer_3, units=64, activation=tf.nn.tanh)
                layer_5 = tf.layers.dense(inputs=layer_4, units=32, activation=tf.nn.tanh)
                self.v_preds = tf.layers.dense(inputs=layer_5, units=1, activation=None)

            self.act_deterministic = tf.argmax(self.act_probs, axis=1)
            self.scope = tf.get_variable_scope().name

    def act(self, vecObs, visObs, sess):
        return sess.run([self.act_probs, self.act_deterministic, self.v_preds], feed_dict={self.vector_obs: vecObs, self.visual_obs: visObs})

    def get_action_prob(self, vecObs, visObs, sess):
        return sess.run(self.act_probs, feed_dict={self.vector_obs: vecObs, self.visual_obs: visObs})

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

