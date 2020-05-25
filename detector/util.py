import numpy as np
import os
import sys
from PIL import Image

if sys.version_info[0] < 3:
    raise Exception("Python 3 is required")


def loadEnv(var):
    if var in os.environ:
        return os.environ[var]
    raise Exception("Missing Required Environment Variable - " + var)


def loadEnvOrEmpty(var):
    if var in os.environ:
        return os.environ[var]
    return ''


def cropAndStoreImage(imagePath, cropFilePath):
    image = np.array(Image.open(imagePath))
    box = getBoxToCrop(cropFilePath, image.shape)
    crop_image = image[box[0]:box[2], box[1]:box[3]]
    coverted_image = Image.fromarray(crop_image, 'L')
    saved_path = str(imagePath).replace(".jpg", "-crop.jpg")
    coverted_image.save(saved_path)
    return saved_path


def getBoxToCrop(cropFilePath, shape):
    with open(cropFilePath) as f:
        boxes_str = f.read()
    box = [float(i) for i in eval(boxes_str.split(" - ")[2])]
    width = int(shape[0])
    height = int(shape[1])
    box_xmin = box[0] * width
    box_ymin = box[1] * height
    box_xmax = box[2] * width
    box_ymax = box[3] * height
    return [int(i) for i in (box_xmin, box_ymin, box_xmax, box_ymax)]
