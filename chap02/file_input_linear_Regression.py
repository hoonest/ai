import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

tf.set_random_seed(777)

xy = np.loadtxt('data-01-test-score.csv', delimiter=',', dtype=np.float32)

x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

print(x_data.shape, x_data, len(x_data))
print(y_data.shape, y_data)

X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# hypothesis 가설 함수 정의
hypothesis = tf.matmul(X, W) + b

# cost/loss 함수(비용/손실함수)
cost = tf.reduce_mean(tf.square(hypothesis - Y))


# Minimize : Gradient Desent(미분:기울기)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)


sess = tf.Session()
sess.run(tf.global_variables_initializer())


for step in range(8001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X:x_data, Y: y_data})

    if step % 100 == 0:
        print(step, "Cost:", cost_val, "\nPrediction:", hy_val)


print("Your Score will be ", sess.run(hypothesis, feed_dict={X:[[85, 90, 75]]}))
