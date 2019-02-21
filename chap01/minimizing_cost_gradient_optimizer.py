import tensorflow as tf

tf.set_random_seed(777)

x_data = [1, 2, 3, 4, 5]
y_data = [1, 2, 3, 4, 5]

W = tf.Variable(-3.0)

# 가설 함수
hypothesis = x_data * W

# cost 함수
cost = tf.reduce_mean(tf.square(hypothesis - y_data))

# Minimize : Gradient Desent(미분:기울기)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

for step in range(101):
    print(step, sess.run(W))
    sess.run(train)


