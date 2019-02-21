import tensorflow as tf

tf.set_random_seed(777)

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

# 가설 함수 H(x)
hypothesis = X * W + b

# cost 함수
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, w_val, b_val, _ = sess.run([cost, W, b, train], \
                                    feed_dict={X:[1,2,3], Y:[1,2,3]})

    if step % 20 == 0:
        print(step, cost_val, w_val, b_val)


print(sess.run(hypothesis, feed_dict={X:[5]}))
print(sess.run(hypothesis, feed_dict={X:[2.5]}))
print(sess.run(hypothesis, feed_dict={X:[1.5, 3.5, 4.5]}))


for step in range(2001):
    cost_val, w_val, b_val, _ = sess.run([cost, W, b, train], \
                            feed_dict={X: [1,2,3,4,5,6,7,8,9,10],
                                       Y: [10,15,20,25,30,35,40,45,50,55]})

    if step % 20 == 0:
        print(step, cost_val, w_val, b_val)

print(sess.run(hypothesis, feed_dict={X:[20]}))




