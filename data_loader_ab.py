import scipy
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from skimage.transform import resize
from skimage.io import imsave
from keras.applications.mobilenet import MobileNet
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.applications.mobilenet import preprocess_input
import os

class DataLoader():
    def __init__(self, dataset_name, img_res=(128, 128)):
        self.dataset_name = dataset_name
        self.img_res = img_res
        self.inception = MobileNet(weights='imagenet',include_top=True)
        self.inception.graph = tf.get_default_graph()
        # Image transformer
        #self.datagen = ImageDataGenerator(shear_range=0.2,zoom_range=0.2,rotation_range=20,horizontal_flip=True)        
     

    
    def load_data(self , batch_size = 1):
        
        path = glob('./datasets/test40/*')
        #batch_images = np.random.choice(path, size=batch_size)
        batch_images = path
        imgs = []
        for img_path in batch_images:            
            imgs.append(img_to_array(load_img(img_path)))
        imgs = np.array(imgs, dtype=float)
        random_batch = 1.0/255*imgs
        
        grayscaled_rgb = gray2rgb(rgb2gray(random_batch))
        embed = self.create_inception_embedding(grayscaled_rgb)
        lab_batch = rgb2lab(random_batch)
        X_batch = lab_batch[:,:,:,0]
        X_batch = X_batch.reshape(X_batch.shape+(1,))
        Y_batch = lab_batch[:,:,:,1:] / 128
        return (Y_batch , embed , X_batch)

    def create_inception_embedding(self,grayscaled_rgb):
        grayscaled_rgb_resized = []
        for i in grayscaled_rgb:
            i = resize(i, (224, 224, 3), mode='constant')
            grayscaled_rgb_resized.append(i)
        grayscaled_rgb_resized = np.array(grayscaled_rgb_resized)
        grayscaled_rgb_resized = preprocess_input(grayscaled_rgb_resized)
        with self.inception.graph.as_default():
            embed = self.inception.predict(grayscaled_rgb_resized)
        return embed

    

    def load_batch(self, batch_size=1):
        path = glob('./datasets/%s/*' % (self.dataset_name))

        self.n_batches = int(len(path) / batch_size)

        for i in range(self.n_batches-1):
            batch = path[i*batch_size:(i+1)*batch_size]
            imgs = []
            for img_path in batch:
                imgs.append(img_to_array(load_img(img_path)))
            imgs = np.array(imgs, dtype=float)
            seq_batch = 1.0/255*imgs
        
            grayscaled_rgb = gray2rgb(rgb2gray(seq_batch))
            embed = self.create_inception_embedding(grayscaled_rgb)
            lab_batch = rgb2lab(seq_batch)
            X_batch = lab_batch[:,:,:,0]
            X_batch = X_batch.reshape(X_batch.shape+(1,))
            Y_batch = lab_batch[:,:,:,1:] / 128 
            yield (Y_batch, embed , X_batch)
            
