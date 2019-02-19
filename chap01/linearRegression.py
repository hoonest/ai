import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# tensorflow를 이용한 Linear Regression

tf.set_random_seed(777)

# X and Y data
x_train = [1, 2, 3]
y_train = [1, 2, 3]

# Hypothesis Func. y = Wx + b
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

#X = tf.placeholder(tf.float32)
#Y = tf.placeholder(tf.float32)
# our hypothesis for linear model x * w

# 가설함수 (H(x)) = xW +b
hypothesis = x_train * W + b

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

# Minimise
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# Launch the graph in a session
sess = tf.Session()

sess.run(tf.global_variables_initializer())

for step in range(2000):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))


