import tensorflow as tf
import numpy as np


tf.set_random_seed(777)

x_data = [[1, 2, 1, 1], [2, 1, 3, 2], [3, 1, 3, 4], [4, 1, 5, 5], [1, 7, 5, 5], [1, 2, 5, 6], [1, 6, 6, 6], [1, 7, 7, 7]]
y_data = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]]

x_test = [[1,2,2,2], [2,7,5,5], [1,8,8,8]]
y_test = [[0,0,1],[0,1,0], [1,0,0]]

X = tf.placeholder("float", [None, 4])
Y = tf.placeholder("float", [None, 3])

nb_classes = 3

W = tf.Variable(tf.random_normal([4, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

# hypothesis 함수
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

# Cross entry cost/loss
cost = tf.reduce_mean(-tf.reduce_sum(Y* tf.log(hypothesis), axis=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(10001):
        sess.run(train, feed_dict={X:x_data, Y:y_data})
        if step % 200 == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={X:x_data, Y:y_data})
            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(step, loss, acc))
            print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}))

    a = sess.run(hypothesis, feed_dict={X:x_test})
    print(a)

    pred, accr = sess.run([prediction, accuracy], feed_dict={X:x_test, Y:y_test})
    print(pred, accr)
    #for p, y in zip(pred, y_data.):
    #    print("[{}] prediction: {} true Y: {}".format(p==int(y), p, int(y)))




