import tensorflow as tf
import numpy as np

x = tf.Variable(0, name='x') #create constant in tf

model = tf.initialize_all_variables() #Stores into model

with tf.Session() as session: #Creates session in tf graph
    for i in range(5):      #loop and add numbers up
        session.run(model)  #Compute variables saved
        x = x + 1           # increment
        print(session.run(x))   #Print
