from scipy import io
import os
import tensorflow as tf
import PIL
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical
class Cifar10Data:
    def loadData(self,num):
        (cifarx_train, cifary_train), (cifarx_test, cifary_test) = cifar10.load_data()

        cifarx_test_process = cifarx_test.astype('float32')/255
        cifary_test = to_categorical(cifary_test)
        np.random.seed(100)
        random_arr = np.random.choice(10000, num, replace=False)
        x_test=[]
        y_test=[]
        imgs=[]
        for randint in random_arr:
            imgs.append(PIL.Image.fromarray(cifarx_test[randint].astype('uint8')))
            x_test.append(cifarx_test_process[randint])
            y_test.append(np.argmax(np.array(cifary_test[randint])))
        return imgs,x_test,y_test
