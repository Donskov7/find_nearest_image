import numpy as np

from keras.models import Model
from keras.layers import Input
from keras.optimizers import SGD
from keras.applications.vgg16 import VGG16


class VGG(object):
    def __init__(self):
        input_ = Input(shape=(224,224,3))
        vgg = VGG16(include_top=True, weights='imagenet', input_tensor=input_)
        self.model = Model(inputs=input_, outputs=vgg.layers[-2].output)
        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(optimizer=sgd, loss='categorical_crossentropy')

    def apply(self, img):
        img[:,:,0] -= 103.939
        img[:,:,1] -= 116.779
        img[:,:,2] -= 123.68
        print(np.shape(img))
        img = img.transpose((1,2,0))
        img = np.expand_dims(img, axis=0)
        return self.model.predict([img])
