import scipy
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from skimage.transform import resize
from skimage.io import imsave

class DataLoader():
    def __init__(self, dataset_name, img_res=(128, 128)):
        self.dataset_name = dataset_name
        self.img_res = img_res

    def load_data(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "test"
        path = glob('./datasets/%s/*' % (self.dataset_name))

        #batch_images = np.random.choice(path, size=batch_size)
        batch_images = path
        imgs_A = []
        imgs_B = []
        for img_path in batch_images:
            img = self.imread(img_path)

            h, w, _ = img.shape
            
            img_A, img_B = img[:, :, :], img[:, :, :]

            img_A = scipy.misc.imresize(img_A, self.img_res)
            img_B = scipy.misc.imresize(img_B, self.img_res)

            color_me = np.array(img_B, dtype=float)
            gray_me = gray2rgb(rgb2gray(color_me/127.5 -1))

            imgs_A.append(img_A)
            imgs_B.append(gray_me)
        
        imgs_A = np.array(imgs_A)/127.5 - 1
        imgs_B = np.array(imgs_B)

        return imgs_A, imgs_B

    def load_batch(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "val"
        path = glob('./datasets/%s/*' % (self.dataset_name))

        self.n_batches = int(len(path) / batch_size)

        for i in range(self.n_batches-1):
            batch = path[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B = [], []
            for img_path in batch:
                img = self.imread(img_path)

                h, w, _ = img.shape

                img_A, img_B = img[:, :, :], img[:, :, :]

                img_A = scipy.misc.imresize(img_A, self.img_res)
                img_B = scipy.misc.imresize(img_B, self.img_res)

                color_me = np.array(img_B, dtype=float)
                gray_me = gray2rgb(rgb2gray(color_me/127.5 -1))

                imgs_A.append(img_A)
                imgs_B.append(gray_me)

            imgs_A = np.array(imgs_A)/127.5 - 1
            imgs_B = np.array(imgs_B)


            yield imgs_A, imgs_B


    def imread(self, path):
        return scipy.misc.imread(path, mode='RGB').astype(np.float)
