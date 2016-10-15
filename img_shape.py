import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

filename = "MarshOrchid.jpg"
image = mpimg.imread(filename)

print(image.shape) #return shape of image matrix
                    #Returns 56528 px high, 3685 px wide and 3 colors deep
plt.imshow(image)
plt.show() #physically shows you image and dimensions!
