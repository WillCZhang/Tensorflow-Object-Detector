import numpy as np
import os
import pathlib
import sys
from PIL import Image, ImageDraw

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
    image = np.array(loadImage(imagePath))
    box = getBoxToCrop(cropFilePath, image.shape)
    crop_image = image[box[1]:box[3], box[0]:box[2]]
    saved_path = str(imagePath).replace(".jpg", "-crop.jpg")
    saveImage(Image.fromarray(crop_image, 'L'), saved_path)
    return saved_path


def recoverCropImageFromPath(path, confidence=0):
    files = [str(p) for p in pathlib.Path(path).glob("*-crop.jpg")]
    extras = [str(p) for p in pathlib.Path(path).glob("*-crop-crop.jpg")]
    files = [f for f in files if f not in extras]
    for i in range(len(files)):
        prefix = str(files[i]).replace("-crop.jpg", "")
        print(f"processing {prefix} - {i + 1} out of {len(files)}")
        recoverCropImage(f'{prefix}.jpg', f'{prefix}.crop',
                         f'{prefix}-crop.result', confidence)


def recoverCropImage(originalImagePath, originalCropFilePath, toRecoverResultFilePath, confidence=0):
    if (not os.path.exists(originalImagePath) or
        not os.path.exists(originalCropFilePath) or
            not os.path.exists(toRecoverResultFilePath)):
        print(
            f"Warning! one of {originalImagePath}, {originalCropFilePath}, and {toRecoverResultFilePath} doesn't exist")
        return
    img = loadImage(originalImagePath)
    recoveredImg = Image.new("RGB", img.size)
    recoveredImg.paste(img)
    originalImage = np.array(img)
    originalBox = getBoxToCrop(originalCropFilePath, originalImage.shape)
    with open(toRecoverResultFilePath) as f:
        results = f.read()
    newBoxes = []
    for result in results.split("\n"):
        if result == '':
            print("Warning! Result file maybe empty" + toRecoverResultFilePath)
            continue
        if parseDetectionScore(result) < confidence:
            continue
        width = originalBox[2] - originalBox[0]
        height = originalBox[3] - originalBox[1]
        objBox = parseDetectionBox(result, [height, width])
        offset = [originalBox[0], originalBox[1],
                  originalBox[0], originalBox[1]]
        # recover coordinators
        objBox = list(np.array(offset) + np.array(objBox))
        offset = [originalImage.shape[0], originalImage.shape[1],
                  originalImage.shape[0], originalImage.shape[1]]
        drawBox(recoveredImg, objBox)
        # nomalize new box
        objBox = list(np.array(objBox) / np.array(offset))
        newBoxes.append(overwriteDetectionBox(result, objBox))
    recoveredPathName = originalImagePath.replace(".jpg", "-recovered.result")
    with open(recoveredPathName, "w") as f:
        f.write("\n".join(newBoxes))
    saveImage(recoveredImg, recoveredPathName.replace(".result", ".jpg"))


def loadImage(path):
    return Image.open(path)


def saveImage(image, dst):
    image.save(dst)


def getBoxToCrop(cropFilePath, shape):
    # This function assumes exactly one line of box info in the crop file
    # shape should be [height, width] -- too lazy to fix naming
    with open(cropFilePath) as f:
        boxes_str = f.read()
    return parseDetectionBox(boxes_str, shape)


def parseDetectionScore(box_str):
    return float(box_str.split(" - ")[1])


def parseDetectionBox(box_str, shape):
    box = [float(i) for i in eval(box_str.split(" - ")[2])]
    width = int(shape[0])
    height = int(shape[1])
    box_xmin = box[0] * width
    box_ymin = box[1] * height
    box_xmax = box[2] * width
    box_ymax = box[3] * height
    return [int(i) for i in (box_ymin, box_xmin, box_ymax, box_xmax)]


def overwriteDetectionBox(box_str, newBox):
    return f'{box_str.split(" - ")[0]} - {box_str.split(" - ")[1]} - {",".join([str(b) for b in newBox])}'


def drawBox(base_image, box):
    new_image = ImageDraw.Draw(base_image)
    new_image.rectangle(box, outline="#79ff68", width=3)
