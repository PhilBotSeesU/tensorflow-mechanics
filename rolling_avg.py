import tensorflow as tf
import numpy as np

data = np.random.randint(1000, size=10000)

x = tf.Variable(0, name='x') #Create variable

model = tf.initialize_all_variables()

with tf.Session() as session:
    for i in range(10):
        session.run(model)
        x = x + data / i
        print(session.run(x))
