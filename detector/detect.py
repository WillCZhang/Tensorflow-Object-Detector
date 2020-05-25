
import numpy as np
import os
import pathlib
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import util
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

# Assume this script will be executed inside docker
model_path = "/model/saved_model/saved_model"
to_detect_path = "/data/to_detect"
label_map_path = '/data/training/label_map.pbtxt'
isGrayscale = bool(util.loadEnv("IS_GRAYSCALE"))

if not os.path.exists(model_path):
    raise IOError(
        'Please save your model before run detect! Try to run "save"')

if not os.path.exists(to_detect_path):
    raise IOError(
        'Please put images you want to detect under your <DATA_PATH>/to_detect')

if not os.path.exists(label_map_path):
    raise IOError(
        'Please make sure you have label_map.pbtxt under <DATA_PATH>/training')

PATH_TO_TEST_IMAGES_DIR = pathlib.Path(to_detect_path)
TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))


category_ids = label_map_util.get_label_map_dict(label_map_path)
needed_boxes = util.loadEnvOrEmpty("NEEDED_BOXES")
needed_boxes_dict = {}
if needed_boxes != '':
    needed_boxes = needed_boxes.replace("\"", "")
    needed_boxes = needed_boxes.replace(" ", "")
    for request in needed_boxes.split(","):
        needed_boxes_dict[category_ids[request.split(":")[0]]] = int(request.split(":")[
            1])

confidence = util.loadEnvOrEmpty("THRESHOLD").replace("\"","")
confidence = 0 if confidence == '' else float(confidence)

def detection_box_format(detection_class, detection_score, detection_box):
    return f"{detection_class} - {detection_score} - {','.join([str(b) for b in detection_box])}"


def load_model():
    model = tf.saved_model.load(model_path)
    model = model.signatures['serving_default']
    return model


def run_inference_for_single_image(model, image, isGrayscale=False):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    if isGrayscale:
        input_tensor = tf.reshape(
            input_tensor, (input_tensor.shape[0], input_tensor.shape[1], 1))
        input_tensor = tf.image.grayscale_to_rgb(input_tensor)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    output_dict = model(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(
        np.int64)

    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                           tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict


def show_inference(model, image_path, isGrayscale=False):
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = np.array(Image.open(image_path))
    output_dict = run_inference_for_single_image(model, image_np, isGrayscale)
    result = []
    boxes = []
    for i in range(len(output_dict['detection_boxes'])):
        label_id = output_dict['detection_classes'][i]
        confidence_score = output_dict['detection_scores'][i]
        # Assume detections are sorted by score
        if float(confidence_score) < confidence:
            break
        formated_result = detection_box_format(
            label_id, confidence_score,
            output_dict['detection_boxes'][i])
        result.append(formated_result)
        if label_id in needed_boxes_dict and needed_boxes_dict[label_id] > 0:
            boxes.append(formated_result)
            needed_boxes_dict[label_id] = needed_boxes_dict[label_id] - 1
    result = "\n".join(result)
    tf.io.write_file(str(image_path).replace("jpg", "result"), result)
    boxes = "\n".join(boxes)
    tf.io.write_file(str(image_path).replace("jpg", "crop"), boxes)
    util.cropAndStoreImage(image_path, str(image_path).replace("jpg", "crop"))

    # Visualization of the results of a detection.
    # vis_util.visualize_boxes_and_labels_on_image_array(
    #     image_np,
    #     output_dict['detection_boxes'],
    #     output_dict['detection_classes'],
    #     output_dict['detection_scores'],
    #     category_index,
    #     instance_masks=output_dict.get('detection_masks_reframed', None),
    #     use_normalized_coordinates=True,
    #     line_thickness=8)


for image_path in TEST_IMAGE_PATHS:
    show_inference(load_model(), image_path, isGrayscale)
