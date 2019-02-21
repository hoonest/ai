import tensorflow as tf

tf.set_random_seed(777)

x_data = [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]]
y_data = [ [0], [0],  [0],  [1],  [1],  [1] ]

X = tf.placeholder(tf.float32, shape=[None,2])
Y = tf.placeholder(tf.float32, shape=[None,1])

W = tf.Variable(tf.random_normal([2,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Logistic hypothesis 함수
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

# cost 함수
cost = -tf.reduce_mean(Y*tf.log(hypothesis)+(1-Y)*tf.log(1-hypothesis))

# 경사하강법 알고리즘
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

predict = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        cost_val, _ =sess.run([cost,train], feed_dict={X:x_data, Y:y_data})

        if step % 200 == 0:
            print(step, cost_val)

    h, c, a = sess.run([hypothesis, predict, accuracy], feed_dict={X:x_data, Y:y_data})
    print('\nHypothesis:',h,'\nPredict:', c,'\nAccuracy:', a)

    print('3,3 시험통과여부: ', sess.run(predict, feed_dict={X:[[3, 3]]}))
    print('3,4 시험통과여부: ', sess.run(predict, feed_dict={X:[[3, 4]]}))
    print('4,0 시험통과여부: ', sess.run(predict, feed_dict={X:[[4, 0]]}))
    print('4,1 시험통과여부: ', sess.run(predict, feed_dict={X:[[4,1]]}))