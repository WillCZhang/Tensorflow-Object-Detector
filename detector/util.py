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
    crop_image = image[box[1]:box[3], box[0]:box[2]]
    coverted_image = Image.fromarray(crop_image, 'L')
    saved_path = str(imagePath).replace(".jpg", "-crop.jpg")
    coverted_image.save(saved_path)
    return saved_path


def getBoxToCrop(cropFilePath, shape):
    # This function assumes exactly one line of box info in the crop file
    # shape should be [height, width] -- too lazy to fix naming
    with open(cropFilePath) as f:
        boxes_str = f.read()
    box = [float(i) for i in eval(boxes_str.split(" - ")[2])]
    width = int(shape[0])
    height = int(shape[1])
    box_xmin = box[0] * width
    box_ymin = box[1] * height
    box_xmax = box[2] * width
    box_ymax = box[3] * height
    return [int(i) for i in (box_ymin, box_xmin, box_ymax, box_xmax)]
