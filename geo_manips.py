import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

filename = "MarshOrchid.jpg"        #filename path
image = mpimg.imread(filename)      #Read image

x = tf.Variable(image, name='x')    #Create a TensFlow variables

model = tf.initialize_all_variables()

with tf.Session() as session:
    x = tf.transpose(x, perm=[0, 1, 2]) #swapping axes around
    session.run(model)                  #Compute
    result = session.run(x)             #store for later usage

plt.imshow(result) #print manipulated image
plt.show() #show
