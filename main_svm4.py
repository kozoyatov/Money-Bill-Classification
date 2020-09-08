import keras
import cv2
import numpy as np
from keras.applications import MobileNetV2
import sklearn.svm as svm
import matplotlib.pylab as plt
import cv2
from utils import prepar_image
import pickle
from pathes import *
from modeClass import *

model = MobileNetV2(weights='imagenet', include_top=False)
lower_blue = np.array([80, 0, 0])
upper_blue = np.array([95, 255, 255])

def train_test_svm(list_of_class_images_train,list_of_class_images_test,mode):

    # svm classifier for each class:
    lin_clf_20 = svm.LinearSVC(max_iter=1000, C=0.1, tol=1e-6)
    lin_clf_50 = svm.LinearSVC(max_iter=1000, C=0.1, tol=1e-6)
    lin_clf_100 = svm.LinearSVC(max_iter=1000, C=0.1, tol=1e-6)
    lin_clf_200 = svm.LinearSVC(max_iter=1000, C=0.1, tol=1e-6)

    bills = [20, 50, 100, 200]

    #################
    ##### TRAIN #####
    #################
    if mode == Modes.train:
        class_features = [1, 2]
        image_index = 0

        # create feature vector for each train image
        for image in list_of_class_images_train:
            print(image_index)
            try:
                image_np = plt.imread(image.path)
            except:
                if image.path[-2] == 'e':
                    image.path = image.path[:-2] + 'g'
                else:
                    image.path = image.path[:-1] + 'e' + 'g'
                image_np = plt.imread(image.path)

            # create mask image for neutralizing similar hue in 50 and 200
            hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
            mask = cv2.inRange(hsv, lower_blue, upper_blue)
            masked_image = np.copy(image_np)
            masked_image[mask != 0] = [0, 0, 0]

            # preper image with image net normalization:
            x = prepar_image(masked_image)
            # x = prepar_image(image_np)

            # create feature vector:
            features = model.predict(x)
            features = np.array(features, dtype=np.float64)
            features = features.reshape((1, 62720))

            # create list of features vectors:
            if image_index <= 1:
                class_features[image_index] = features
                if image_index == 1:
                    class_features = np.concatenate((class_features[0], class_features[1]), axis=0)
                image_index = image_index + 1
                continue
            class_features = np.concatenate((class_features, features), axis=0)
            image_index = image_index + 1

        # train labels:
        y_20 = []
        y_50 = []
        y_100 = []
        y_200 = []

        #creating train labels list:
        for image in list_of_class_images_train:
            # y_20
            if (image.data == 0):
                y_20.append(1)
            else:
                y_20.append(0)

            # y_50
            if (image.data == 1):
                y_50.append(1)
            else:
                y_50.append(0)

            # y_100
            if (image.data == 2):
                y_100.append(1)
            else:
                y_100.append(0)

            # y_200
            if (image.data == 3):
                y_200.append(1)
            else:
                y_200.append(0)

        y_20 = np.array(y_20)
        y_50 = np.array(y_50)
        y_100 = np.array(y_100)
        y_200 = np.array(y_200)
        print('start fitting')
        # fit 4 svm models:
        print(20)
        lin_clf_20.fit(class_features, y_20)
        print(50)

        lin_clf_50.fit(class_features, y_50)
        print(100)

        lin_clf_100.fit(class_features, y_100)
        print(200)

        lin_clf_200.fit(class_features, y_200)

        # save models weights:
        pickle.dump(lin_clf_20, open(pickle_20_path, 'wb'))
        pickle.dump(lin_clf_50, open(pickle_50_path, 'wb'))
        pickle.dump(lin_clf_100, open(pickle_100_path, 'wb'))
        pickle.dump(lin_clf_200, open(pickle_200_path, 'wb'))

    ################
    ##### TEST #####
    ################
    print('start testing')

    # Heat map params:
    predictions_heat_map = []
    y_heat_map = []

    if mode == Modes.test:
        # loading models
        lin_clf_20 = pickle.load(open(pickle_20_path, 'rb'))
        lin_clf_50 = pickle.load(open(pickle_50_path, 'rb'))
        lin_clf_100 = pickle.load(open(pickle_100_path, 'rb'))
        lin_clf_200 = pickle.load(open(pickle_200_path, 'rb'))

    num_of_test_images = float(0)
    num_of_right_test_images = float(0)

    #creating test labels list:
    for image in list_of_class_images_test:
        try:
            image_np = plt.imread(image.path)
        except:
            if image.path[-2]=='e':
                image.path = image.path[:-2]+'g'
            else:
                image.path=image.path[:-1]+'e'+'g'
            image_np = plt.imread(image.path)

        # creating train labels list:
        hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        masked_image = np.copy(image_np)
        masked_image[mask != 0] = [0, 0, 0]

        # creating train labels list:
        x = prepar_image(masked_image)
        features1 = model.predict(x)
        features1 = np.array(features1, dtype=np.float64)
        features1 = features1.reshape((1, 62720))

        # prediction for each class:
        tmp_20 = lin_clf_20._predict_proba_lr(features1)
        tmp_50 = lin_clf_50._predict_proba_lr(features1)
        tmp_100 = lin_clf_100._predict_proba_lr(features1)
        tmp_200 = lin_clf_200._predict_proba_lr(features1)

        tmp = [tmp_20[0][1], tmp_50[0][1], tmp_100[0][1], tmp_200[0][1]]
        if tmp_50[0][1] > 0.45 and tmp_200[0][1] < 0.6 and tmp_20[0][1] < 0.5 and tmp_100[0][1] < 0.5:
            predicted = 1
            print('\n%%%%%%%%%%%%%%%%%%%%%%%\n')
            print('50: ' + str(tmp_50[0][1]))
            print('2000: ' + str(tmp_200[0][1]))
            print('\n%%%%%%%%%%%%%%%%%%%%%%%\n')

        else:
            predicted = np.argmax(tmp)

        # Update heat map params
        predictions_heat_map.append(predicted)
        y_heat_map.append(image.data)

        print('\n\n-------------------------------\n\n')

        if predicted == image.data:
            num_of_right_test_images = num_of_right_test_images + float(1)

        else:
            print('--------------------')
            print('------ FAILED ------')
            print('--------------------\n\n')

        num_of_test_images = num_of_test_images+float(1)
        print("real:" + str(bills[image.data]), "pred:" + str(bills[predicted]))
        print("20: " + str(tmp_20[0]))
        print("50: " + str(tmp_50[0]))
        print("100: " + str(tmp_100[0]))
        print("200: " + str(tmp_200[0]))
        print(f"iter num {num_of_test_images} : {num_of_right_test_images/num_of_test_images}")


    return None


