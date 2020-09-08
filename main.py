from main_svm4 import train_test_svm
from predict_from_cam import predict_from_cam
from imageClass import *
from createImafeList import open_folder_list
from pathes import base_path_train_txt
from pathes import base_path_test_txt


def main():

    mode = Modes.train

    if mode == Modes.test or mode == Modes.train:
        list_of_class_images_train = open_folder_list(base_path_train_txt, Type.train)
        list_of_class_images_test = open_folder_list(base_path_test_txt, Type.test)
        train_test_svm(list_of_class_images_train, list_of_class_images_test, mode)

    if mode == Modes.camera:
        predict_from_cam()
    return


if __name__ == "__main__":
    main()























# # original code with prob.
# import keras
# import cv2
# import numpy as np
# from keras.applications import MobileNetV2
# from keras.models import Model
# import sklearn
# import sklearn.svm as svm
# from math import exp
# import matplotlib.pylab as plt
# import sys
# base_path_train="C:\\Users\\Ilan\\Desktop\\dataset\\224224\\single224224\\train\\"
# base_path_test="C:\\Users\\Ilan\\Desktop\\dataset\\224224\\single224224\\test\\"
#
# class_features=[1,2]
#
# x=plt.imread(base_path_train+'single_train (1).jpeg')
# x = np.expand_dims(x, axis=0)
# model=MobileNetV2(weights='imagenet')
# model = Model(inputs=model.input, outputs=model.layers[-2].output)
# features=model.predict(x)
# features=np.array(features,dtype=np.float64)
# class_features[0]=features
#
# x=plt.imread(base_path_train+'single_train (2).jpeg')
# x = np.expand_dims(x, axis=0)
# features1=model.predict(x)
# features1=np.array(features1,dtype=np.float64)
#
# class_features=np.concatenate((features,features1),axis=0)
#
#
# for i in range(3, 63):  # change while to for the image
#     print(i)
#     image_np = plt.imread(base_path_train+'single_train '+'('+ str(i) +')'+ ".jpeg")
#     x= image_np
#     x = np.expand_dims(x, axis=0)
#     features1 = model.predict(x)
#     class_features = np.concatenate((class_features, features1), axis=0)
#
# y=[3,1,3,3,0,3,0,3,3,0,1,2,1,0,1,0,1,3,3,3,3,1,0,1,3,1,0,3,1,1,1,1,1,1,1,0,0,1,0,0,0,0,0,0,0,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2]
# lin_clf = svm.LinearSVC()
# lin_clf.fit(class_features,y)
#
#
# image_index=[]
# pred_list=[]
# for i in range(1,19):
#     image_np = plt.imread(base_path_test+'single_test '+'('+ str(i) +')'+ ".jpeg")
#     x = image_np
#     x = np.expand_dims(x, axis=0)
#     features1 = model.predict(x)
#     image_index.append(i)
#     tmp=lin_clf.decision_function(features1)
#     softmax=[np.exp(tmp[0][i])/np.sum(np.exp(tmp)) for i in range(4)]
#     tmp=lin_clf.predict(features1)
#     pred_list.append(tmp)
#     print(i,softmax)
#
# fig,axes=plt.subplots(nrows=2,ncols=3,figsize=(10,10))
# ax=axes.ravel()
# for i in range(6):
#   image_np = plt.imread(str(image_index[i]) + "000.jpeg")
#   if(pred_list[i][0]==0):
#     string="20 BILL"
#   if(pred_list[i][0]==1):
#     string="50 BILL"
#   if(pred_list[i][0]==2):
#     string="100 BILL"
#   if(pred_list[i][0]==3):
#     string="200 BILL"
#   ax[i].imshow(image_np)
#   ax[i].set_title("{} ".format(string))
# plt.tight_layout()
# plt.show()
#
#
