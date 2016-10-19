import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

filename = "MarshOrchid.jpg"
image = mpimg.imread(filename)
height, width, depth = image.shape #feeds variables respective to return order

x = tf.Variable(image, name='x')

model = tf.initialize_all_variables()

with tf.Session() as session:
    x = tf.reverse_sequence(x, [width] * height, 1, batch_dim = 0) #swap pizels from left to right top row to bottom order
    session.run(model)
    result = session.run(x)

print(result.shape) #shape reversed now
plt.imshow(result)
plt.show()
