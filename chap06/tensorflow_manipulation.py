# Tensor Manipulation
# https://www.tensorflow.org/api_docs/python/

import tensorflow as tf
import numpy as np
import pprint

tf.set_random_seed(777)

pp = pprint.PrettyPrinter(indent=4)
sess = tf.InteractiveSession()

# 1차원 배열
t = np.array([0.,1.,2.,3.,4.,5.,6.])
pp.pprint(t)
print(t)
print(t.shape) # shape, (7,)
print(t.ndim)  # rank, 1
print(t[0],t[1],t[-1]) # 0.0 1.0 6.0
print(t[2:5], t[4:-1]) # [2. 3. 4.] [4. 5.]
print(t[:2], t[3:]) # [0. 1.] [3. 4. 5. 6.]

# 2차원 배열(Matrix:행렬)
t = np.array([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.],[10.,11.,12.]]) # 4x3행렬
pp.pprint(t)
print(t.ndim)  # rank-2
print(t.shape) # shape(4,3)

# shape, Rank, Axis
t = tf.constant([1,2,3,4])
print(tf.shape(t)) # Tensor("Shape:0", shape=(1,), dtype=int32)
print(tf.shape(t).eval()) # [4]

t = tf.constant([[1,2],[3,4]])
print(tf.shape(t)) # Tensor("Shape_2:0", shape=(2,), dtype=int32)
print(tf.shape(t).eval()) # [2 2]

t = tf.constant([[[1,2,3,4],[5,6,7,8]],
                 [[9,10,11,12],[13,14,15,16]],
                 [[17,18,19,20],[21,22,23,24]]])
print(tf.shape(t)) # Tensor("Shape_4:0", shape=(3,), dtype=int32)
print(tf.shape(t).eval()) # [3 2 4]










