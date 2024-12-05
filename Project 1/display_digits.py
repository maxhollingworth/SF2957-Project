# Display images from the digits data set

import matplotlib.pyplot as plt
from sklearn import datasets

# import some data to play with
digits = datasets.load_digits()
print(digits[0,:])

im = digits.images[1]
print(im.shape)
plt.gray()
for i in range(1,10):
    plt.matshow(digits.images[i])
plt.show() 
