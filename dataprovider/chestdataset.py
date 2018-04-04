import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

class ChestDataSet:
    trainData = []
    trainLabel = []
    testData = []
    testLabel = []

    def __init__(self):
        self.data_dir = 'dataset/'
        self.train_dir = 'dataset/train/'
        self.dev_dir = 'dataset/dev/'
        self.test_dir = 'dataset/test/'

        self.num_train_samples = len(os.listdir(self.train_dir))
        self.num_dev_samples = len(os.listdir(self.dev_dir))
        self.num_test_samples = len(os.listdir(self.test_dir))
        self.width = 64
        self.height = 64
        #self.num_dev_samples = 100
        #self.num_test_samples = 100
        #self.num_train_samples = 100


        self.x_train = np.zeros((self.num_train_samples, self.width, self.height, 3), dtype='uint8')
        self.y_train = np.zeros((self.num_train_samples, 1), dtype='uint8')
        self.x_dev = np.zeros((self.num_dev_samples, self.width, self.height, 3), dtype='uint8')
        self.y_dev = np.zeros((self.num_dev_samples, 1), dtype='uint8')
        self.x_test = np.zeros((self.num_test_samples, self.width, self.height, 3), dtype='uint8')
        self.y_test = np.zeros((self.num_test_samples, 1), dtype='uint8')
        self.test_names = []


    def whiteImage(self, img):
        meanValue, stddevValue = cv2.meanStdDev(img)
        img2 = np.array(img, dtype=np.float64)
        img2 = img - meanValue[0]
        img2 = img2 / stddevValue[0]
        return img2
       
    def loadDataset(self, bWhiteImage = True):
        if (not os.path.exists(self.data_dir)):
           raise Exception('failed to open directory')
        train_file_list = os.listdir(self.train_dir)
        dev_file_list = os.listdir(self.dev_dir)
        test_file_list = os.listdir(self.test_dir)

        for idx, file_name in enumerate(train_file_list):
            #if idx >= 100:
            #    break
            img = cv2.imread(self.train_dir + file_name)
            img = cv2.resize(img, (self.width, self.height))
            if bWhiteImage:
                img = self.whiteImage(img)
            self.x_train[idx,:,:,:] = img[:, :, :]
            self.y_train[idx, :] = int(file_name[0])
            

        for idx, file_name in enumerate(dev_file_list):
            #if idx >= 100:
            #    break
            img = cv2.imread(self.dev_dir + file_name)
            img = cv2.resize(img, (self.width, self.height))
            if bWhiteImage:
                img = self.whiteImage(img)
            self.x_dev[idx,:,:,:] = img
            self.y_dev[idx,:] = int(file_name[0])

        for idx, file_name in enumerate(test_file_list):
            #if idx >= 100:
            #    break
            img = cv2.imread(self.test_dir + file_name)
            img = cv2.resize(img, (self.width, self.height))
            if bWhiteImage:
                img = self.whiteImage(img)
            self.x_test[idx,:,:,:] = img; 
            self.y_test[idx,:] = int(file_name[0])
            self.test_names.append(file_name)
        print('shape of x_train', self.x_train.shape)
        print('shape of y_train', self.y_train.shape)
        print('max of y_train:', np.max(self.y_train))
        print('shape of x_dev', self.x_dev.shape)
        print('shape of y_dev', self.y_dev.shape)
        print('shape of x_test', self.x_test.shape)
        print('shape of y_test', self.y_test.shape)

        return (self.x_train, self.y_train), (self.x_dev, self.y_dev), (self.x_test, self.y_test, self.test_names)


if __name__ == '__main__':
        data = ChestDataSet()
        (x_train, y_train), (x_dev, y_dev), (x_test, y_test) = data.loadDataset(bWhiteImage = False)
        plt.imshow(x_train[0,:,:,:], cmap=plt.cm.gray)
        plt.show()
        input()
    
