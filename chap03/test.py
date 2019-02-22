import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

tf.set_random_seed(777)

xy = np.loadtxt('data-03-diabetes.csv', delimiter=',', dtype=np.float32)

#x_data = xy[:, 0:-1]
#y_data = xy[:, [-1]]

x_data = xy[:749, 0:-1]
y_data = xy[:749, [-1]]

x_test = xy[749:, 0:-1]
y_test = xy[749:, [-1]]

print(x_data.shape, y_data.shape)
print(x_test.shape, y_test.shape)

X = tf.placeholder(tf.float32, shape=[None,8])
Y = tf.placeholder(tf.float32, shape=[None,1])

W = tf.Variable(tf.random_normal([8,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Logistic hypothesis 함수
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)
#hypothesis = tf.nn.selu(tf.matmul(X, W) + b)



# cost 함수
cost = -tf.reduce_mean(Y*tf.log(hypothesis)+(1-Y)*tf.log(1-hypothesis))
#cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis)))
#cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))


# 경사하강법 알고리즘
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

predict = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    cost_arr = []
    step_arr = []


    for step in range(20001):
        cost_val, _ = sess.run([cost, train], feed_dict={X:x_data, Y:y_data})

        cost_arr.append(cost_val)
        step_arr.append(step)

        if step % 1000 == 0:
            print(step, cost_val)

    h, c, a = sess.run([hypothesis, predict, accuracy], feed_dict={X:x_data, Y:y_data})
    print('\nHypothesis:', h ,'\nPredict:', c ,'\nAccuracy:', a)

    print('예측시험통과여부: ', sess.run(predict, feed_dict={X: x_test}) )

    ac = sess.run([accuracy], feed_dict={X: x_test, Y: y_test})
    print('실제시험통과률 : ', ac)

    plt.plot(step_arr, cost_arr)
    plt.show()
