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

