import cv2
import numpy as np
from imageClass import *


# get path to images and return Image Class list

def open_folder_list(path, whatType=Type):

    try:
        pathFolderList = path + "/list.txt"
        with open(pathFolderList) as f:
            content = f.readlines()
        content = [x.strip() for x in content]

    finally:
        f.close()

    image_list = []

    for x in content:
        folderName = x[0:4]
        numOfImages=x[5:]
        folderPath = path + "\\" + folderName + "\\"

        for imageIndex in range(0,int(numOfImages)):
            imagePath = folderPath + folderName + " " + "(" + str(imageIndex + 1) + ").jpg"
            image = Image(imagePath,folderName)
            image.setData()
            image.type = whatType
            image_list.append(image)

    return image_list

