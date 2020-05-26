import io
import math
import os
import random
import shutil
import tensorflow as tf
import time
from lxml import etree

import dataset_util
import util

# Configurations (in most cases you don't need to change them)
ratio = 0.2  # Percentage of the total images used for testing, between 0~1
labelMapFilename = "label_map.pbtxt"  # Corresponding to pipeline.config
recordFilename = "tf.record"  # Corresponding to pipeline.config

# Assuming this script will be executed inside docker container
imagePath = "/data/images"
trainingPath = "/data/training"
testingPath = "/data/testing"


classes = [className.replace("\"", "").strip()
           for className in util.loadEnv("CLASSES").split(",")]
if len(classes) == 0:
    raise ValueError("Please provide at least one class for training")


class Image:
    def __init__(self, imageName, srcPath=imagePath, dstPath=None):
        def copyOrGetFilePath(filename):
            src = os.path.join(srcPath, filename)
            if dstPath is None:
                return src
            dst = os.path.join(dstPath, filename)
            if os.path.exists(src):
                shutil.copyfile(src, dst)
                return dst
            return ''
        self.jpg = copyOrGetFilePath(imageName+".jpg")
        self.xml = copyOrGetFilePath(imageName+".xml")
        self.crop = copyOrGetFilePath(imageName+".crop")

    def createTFExample(self):
        """Convert XML derived dict to tf.Example proto.
        Notice that this function normalizes the bounding box coordinates provided
        by the raw data.
        Args: None
        Returns:
            example: The converted tf.Example.
        """
        with tf.io.gfile.GFile(self.xml, 'r') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = dataset_util.recursive_parse_xml_to_dict(xml)[
            'annotation']
        # the image might be processed in a different location
        # so overwrite the path to the input image path for consistency
        data['path'] = self.jpg if self.crop == '' else self.__cropImage(data)

        width = int(data['size']['width'])
        height = int(data['size']['height'])
        filename = data['filename'].encode('utf8')
        with tf.io.gfile.GFile(data['path'], 'rb') as fid:
            encoded_image_data = fid.read()
        image_format = 'jpeg'.encode('utf8')

        # List of normalized left x coordinates in bounding box (1 per box)
        xmins = []
        # List of normalized right x coordinates in bounding box (1 per box)
        xmaxs = []
        # List of normalized top y coordinates in bounding box (1 per box)
        ymins = []
        # List of normalized bottom y coordinates in bounding box (1 per box)
        ymaxs = []
        # List of string class name of bounding box (1 per box)
        classes_text = []
        classes_id = []  # List of integer class id of bounding box (1 per box)

        for obj in data['object']:
            if obj['name'] not in classes or not self.__isValidBox(obj, width, height):
                print('Unexpected object: ' + str(obj) + ' in ' + data['path'])
                continue
            xmins.append(float(obj['bndbox']['xmin']) / width)
            ymins.append(float(obj['bndbox']['ymin']) / height)
            xmaxs.append(float(obj['bndbox']['xmax']) / width)
            ymaxs.append(float(obj['bndbox']['ymax']) / height)
            classes_text.append(obj['name'].encode('utf8'))
            classes_id.append(getClassID(obj['name']))

        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(filename),
            'image/source_id': dataset_util.bytes_feature(filename),
            'image/encoded': dataset_util.bytes_feature(encoded_image_data),
            'image/format': dataset_util.bytes_feature(image_format),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes_id),
        }))
        return tf_example

    def __cropImage(self, data):
        # Note: box format is (normalized) ymin, xmin, ymax, xmax
        box = util.getBoxToCrop(
            self.crop, [int(data['size']['height']), int(data['size']['width'])])
        data['size']['width'] = box[3] - box[1]
        data['size']['height'] = box[2] - box[0]
        for obj in data['object']:
            # sadly there's no pointer in python... so can't abstract away
            obj['bndbox']['xmin'] = float(obj['bndbox']['xmin']) - box[0]
            obj['bndbox']['ymin'] = float(obj['bndbox']['ymin']) - box[1]
            obj['bndbox']['xmax'] = float(obj['bndbox']['xmax']) - box[0]
            obj['bndbox']['ymax'] = float(obj['bndbox']['ymax']) - box[1]
        return util.cropAndStoreImage(self.jpg, self.crop)

    def __isValidBox(self, obj, width, height):
        return (0 < obj['bndbox']['xmin'] < width and
                0 < obj['bndbox']['xmax'] < width and
                0 < obj['bndbox']['ymin'] < height and
                0 < obj['bndbox']['ymax'] < height)


def getClassID(className):
    # ID is 1-index
    return classes.index(className) + 1


def generateDataset(labeledImageNames, labelMap, dst):
    with open(os.path.join(dst, labelMapFilename), 'w') as f:
        f.write(labelMap)

    writer = tf.io.TFRecordWriter(os.path.join(dst, recordFilename))

    for imageName in labeledImageNames:
        image = Image(imageName, dstPath=dst)
        tf_example = image.createTFExample()
        writer.write(tf_example.SerializeToString())

    writer.close()


def generateLabelMapItem(className):
    return f"""item {{
    id: {getClassID(className)}
    name: '{className}'
}}
"""


def run():
    # Only process labeled images (i.e. images contain both jpg & xml file)
    labeledImageNames = []
    temp = {''}
    for file in os.listdir(imagePath):
        if file.endswith('.jpg') or file.endswith('.xml'):
            filename = file.split('.')[0]
            if filename in temp:
                labeledImageNames.append(filename)
            else:
                temp.add(filename)

    # Spliting Testing & Training Datasets
    testImageNames = []
    for i in range(math.ceil(ratio * len(labeledImageNames))):
        imageName = labeledImageNames[random.randint(
            0, len(labeledImageNames)-1)]
        testImageNames.append(imageName)
        labeledImageNames.remove(imageName)

    # Generate Label Map
    labelMap = "\n".join([generateLabelMapItem(className)
                          for className in classes])

    generateDataset(labeledImageNames, labelMap, trainingPath)
    generateDataset(testImageNames, labelMap, testingPath)


run()
