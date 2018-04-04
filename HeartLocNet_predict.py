'''predict heart location using a simple deep CNN on the X-CHEST small images dataset.

It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model
import cv2
import os
from shutil import copyfile, move # use to import copyfile
import numpy as np
import sys
sys.path.append('dataprovider/')
from chestdataset import ChestDataSet



'''
predict each image in the directory of test using trained model
'''
if __name__ == '__main__':
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'keras_cifar10_trained_sigmoid_model.h5'
    model_path = os.path.join(save_dir, model_name)
    loaded_mode = load_model(model_path)
    #predict(self, x, batch_size=None, verbose=0, steps=None)
    #test_image_dir = 'dataset/test/'
    test_image_dir = 'dataset/test/'
    file_list = os.listdir(test_image_dir)
    mislabel_dir = 'dataset/mislabel_test/'
    total_num = len(file_list)
    correct_num = 0
    for file_name in file_list:
        img = cv2.imread(test_image_dir + file_name)
        ground_result = int(file_name[0])
        img = cv2.resize(img, (64, 64))
        img0 = img.astype('float32')
        img0 = img0 / 255
        img0 = img0[np.newaxis, :, :, :]
        pred_result = loaded_mode.predict(img0, verbose=1)
        if np.abs(ground_result - pred_result) < 0.5:
            print('success predict file {0} with probability {1}%'.format(file_name, 100*np.max([pred_result, 1 - ground_result - pred_result])))
            correct_num += 1
            fid = open('predict.log', 'a')
            fid.write('success predict file {0} with probability {1}%'.format(file_name, 100*np.max([pred_result, 1 - ground_result - pred_result]))+'\n')
            fid.close()
        else:
            print('fail to calssify file {0}, result is: {1} '.format(file_name, pred_result))
            fid = open('predict.log', 'a')
            fid.write('fail to calssify file {0}, result is: {1} '.format(file_name, pred_result))
            fid.close()
            if not os.path.exists(mislabel_dir):
                os.makedirs(mislabel_dir)
            copyfile(os.path.join(test_image_dir, file_name), os.path.join(mislabel_dir, file_name))
    print('accuracy: {0}'.format(correct_num / total_num))
 



#'''
#run batch evaluate to test data
#'''
#if __name__ == '__main__':
#    save_dir = os.path.join(os.getcwd(), 'saved_models')
#    model_name = 'keras_cifar10_trained_sigmoid_model.h5'
#    model_path = os.path.join(save_dir, model_name)
#    loaded_mode = load_model(model_path)

#    dataset = ChestDataSet()
#    (x_train, y_train), (x_dev, y_dev), (x_test, y_test, _) = dataset.loadDataset(bWhiteImage = False)

#    x_train = x_train.astype('float32')
#    x_dev = x_dev.astype('float32')
#    x_test = x_test.astype('float32')
    

#    x_train /= 255
#    x_dev /= 255
#    x_test /= 255

#    scores = loaded_mode.evaluate(x_test, y_test, verbose=1)
#    print('Test loss:', scores[0])
#    print('Test accuracy:', scores[1]) #0.965


          
        
