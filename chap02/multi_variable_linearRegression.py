import tensorflow as tf

tf.set_random_seed(777)

x_data = [[73.0, 80.0, 75.0],[93.0,88.0, 93.0],[89.0,91.0,90.0],[96.0,98.0, 100.0 ],[73.0,66.0, 70.0 ]]
y_data = [[152.0], [185.0], [180.0], [196.0], [142.0]]

X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')


hypothesis = tf.matmul(X, W) + b

# cost/loss 함수(비용/손실함수)
cost = tf.reduce_mean(tf.square(hypothesis - Y))


# Minimize : Gradient Desent(미분:기울기)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(8001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                                   feed_dict={X:x_data, Y: y_data})

    if step % 100 == 0:
        print(step, "Cost:", cost_val, "\nPrediction:", hy_val)

print("Your Score will be ", sess.run(hypothesis, feed_dict={X:[[85, 90, 75]]}))