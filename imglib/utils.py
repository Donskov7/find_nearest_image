import hashlib
import numpy as np
import pickle
from scipy.spatial.distance import cosine

from sklearn.neighbors import KDTree as KDTree_base
from sklearn.decomposition import PCA as PCA_base

from keras.models import Model
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions


class VGG(object):
    def __init__(self, top=5):
        self.top = top
        self.full_model = VGG16(include_top=True, weights='imagenet')
        self.model = Model(inputs=self.full_model.input, outputs=self.full_model.get_layer('fc2').output)
        print('MODEL was loaded')

    def predict(self, img, full=False):
        img = image.load_img(img, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        if full:
            res = self.full_model.predict([x])
            return decode_predictions(res, top=self.top)[0]
        res = self.model.predict([x])
        return res[0]

    def get_hash(self, vec):
        if vec is None:
            return None
        return hashlib.md5(' '.join([str(val) for val in vec]).encode('utf-8')).hexdigest()


class KDTree(KDTree_base):
    def load(self, fname):
        self = pickle.load(open(fname, 'rb'))
        return self

    def save(self, fname):
        pickle.dump(self, open(fname, 'wb'))
        return self


class PCA(PCA_base):
    def load(self, fname):
        self = pickle.load(open(fname, 'rb'))
        return self

    def save(self, fname):
        pickle.dump(self, open(fname, 'wb'))
        return self


def find_nearest_doc(doc0, all_docs):
    nearest = None
    for doc1 in all_docs:
        sim = cosine(doc0['vec'], doc1['vec'])
        if nearest is None or sim < nearest['sim']:
            nearest = doc1
            nearest['sim'] = sim
    return nearest
