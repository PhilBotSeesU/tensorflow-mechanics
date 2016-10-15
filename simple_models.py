import tensorflow as tf
import numpy as np

z = tf.constant(35, name='z') # this is indepenent
r = tf.Variable(z + 5, name='r') # embodies all constants or some constants into variables
                                # z has a dependency on z

model = tf.initialize_all_variables() # stores all variables which hold the constants

with tf.Session() as session: #Computing variables...
    session.run(model) #Run Tensorflow session
    print(session.run(r)) #run variable desired from model embodiment

print("array exercise")

a = tf.constant([35, 50, 40], name='a')
b = tf.Variable(a + 5, name='b')

model2 = tf.initialize_all_variables()
