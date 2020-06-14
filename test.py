import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from pandas.io.parsers import read_csv

data = read_csv('gold.csv', sep=',')
xy = np.array(data, dtype = np.float32)

x_data = xy[:,0:-1]
y_data = xy[:,[-1]]

x = tf.placeholder(tf.float32, shape=[None, 3])
y = tf.placeholder(tf.float32, shape=[None, 1])
W = tf.Variable(tf.random_normal([3, 1]), name="Weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

hypothesis = tf.matmul(x,W) + b

cost = tf.reduce_mean(tf.square(hypothesis-y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000000006)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(100001):
    cost_, hypo_, _ = sess.run([cost, hypothesis, train], feed_dict={x : x_data, y : y_data})
    if step % 5000 == 0:
        print(step, "손실비용 :", cost_)
        print("-금 시세 : ", hypo_)

saver = tf.train.Saver()
save_path = saver.save(sess, "./svaed.cpkt")
print("학습된 모델을 저장하였습니다.")