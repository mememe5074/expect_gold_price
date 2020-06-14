import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
from pandas.io.parsers import read_csv
import numpy as np

data = read_csv('gold.csv', sep=',')

x = tf.placeholder(tf.float32, shape=[None, 3])
y = tf.placeholder(tf.float32, shape=[None, 1])
W = tf.Variable(tf.random_normal([3, 1]), name="Weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

hypothesis = tf.matmul(x, W) + b

saver = tf.train.Saver()
model = tf.global_variables_initializer()

nas = float(input("나스닥 : "))
oil = float(input("국제유가WTI : "))
usBond = float(input("미국 채권 상승률 : "))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    save_path = tf.train.latest_checkpoint('./')

    saver.restore(sess, save_path)

    data = ((nas, oil, usBond),)
    arr = np.array(data, dtype=np.float32)

    x_data = arr[0:3]
    dict = sess.run(hypothesis, feed_dict={x: x_data})

    print(dict[0])
