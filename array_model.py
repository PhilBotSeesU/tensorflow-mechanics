import tensorflow as tf
import numpy as np

print("Create numpy array 10,000 random numbers into x")

f = tf.constant(np.random.rand(10000),name='f')
g = tf.Variable(f, name='g')

model3 = tf.initialize_all_variables()

with tf.Session() as session:
    session.run(model3)
    print(session.run(g))
