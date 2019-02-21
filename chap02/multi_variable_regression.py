import tensorflow as tf
import matplotlib.pyplot as plt

x1_data = [73.0, 93.0, 89.0, 96.0, 73.0]
x2_data = [80.0, 88.0, 91.0, 98.0, 66.0]
x3_data = [75.0, 93.0, 90.0, 100.0, 70.0]
y_data = [152.0, 185.0, 180.0, 196.0, 142.0]

x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)

Y = tf.placeholder(tf.float32)

w1 = tf.Variable(tf.random_normal([1]), name='weight1')
w2 = tf.Variable(tf.random_normal([1]), name='weight2')
w3 = tf.Variable(tf.random_normal([1]), name='weight3')
b = tf.Variable(tf.random_normal([1]), name='bias')


hypothesis = x1 * w1 + x2 * w2 + x3 * w3 + b

# cost/loss 함수(비용/손실함수)
cost = tf.reduce_mean(tf.square(hypothesis - Y))


# Minimize : Gradient Desent(미분:기울기)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

W_history = []
cost_history = []

for step in range(4001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={x1:x1_data, x2:x2_data, x3:x3_data, Y:y_data})
#    W_history.append(hy_val)
#    cost_history.append(cost_val)

    if step % 10 == 0:
        print(step, 'Cost:', cost_val, '\nprediction:', hy_val)

#plt.plot(W_history, cost_history)
#plt.show()
