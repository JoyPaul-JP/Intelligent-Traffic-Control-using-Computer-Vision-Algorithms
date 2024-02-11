import os
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
# from utils.datasets import letterbox
from utils.general import (non_max_suppression, check_img_size)
from utils.torch_utils import select_device
import numpy as np


device = select_device("0")
imgsz = 640
model = attempt_load("./runs/train/exp2/weights/best.pt", device=device)  # load FP32 model
imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size

# # load model
# def init():
#     # global model
#     # global names
#     # global imgsz
#     # device = select_device("0")
#     # Load model


names = model.module.names if hasattr(model, 'module') else model.names
    
def letterbox(img, new_shape=(imgsz, imgsz), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Compute new size keeping the aspect ratio
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))

    # Compute padding
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh should be a multiple of 64
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    # Add padding to the image
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def scale_coords(img_shape, coords, original_shape):
    """
    Scale bounding box coordinates from a resized image back to the original image size.

    Parameters:
        img_shape (tuple): Shape of the resized image (height, width).
        coords (torch.Tensor): Bounding box coordinates in the format (xmin, ymin, xmax, ymax).
        original_shape (tuple): Shape of the original image (height, width).

    Returns:
        torch.Tensor: Scaled bounding box coordinates.
    """
    ih, iw = img_shape
    oh, ow = original_shape[:2]

    scale = min(ow / iw, oh / ih)
    nw, nh = round(iw * scale), round(ih * scale)
    dx, dy = (ow - nw) / 2, (oh - nh) / 2

    coords[:, [0, 2]] -= dx
    coords[:, [1, 3]] -= dy
    coords[:, [0, 2]] /= scale
    coords[:, [1, 3]] /= scale

    return coords



# get fdetections
def detect_image(img_main, device):
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)

    img = letterbox(img_main, new_shape=imgsz)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)

    img = img.float()  # uint8 to fp16/32
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # pred
    pred = model(img, augment=False)[0]

    # Apply NMS
    pred = non_max_suppression(pred, 0.4, 0.5, classes=None, agnostic=False)


    results = []
    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if det is not None and len(det):
            # Rescale boxes from img_size to img_main size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_main.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                results.append({
                    'xmin' : int(xyxy[0]),
                    'ymin' : int(xyxy[1]),
                    'xmax' : int(xyxy[2]),
                    'ymax' : int(xyxy[3]),
                    'class' : names[int(cls)]
                })
    return results


# read image using opnecv
image = cv2.imread("./data/images/vlcsnap-2023-07-28-16h04m16s537.png")
height, width, _ = image.shape
image = cv2.resize(image, (640,640))
resp = detect_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), device=device) # pass RGB image
# iterate over each detection and draw on image
for bbox in resp:
    xmin, ymin, xmax, ymax = bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']
    xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)  # Scale the coordinates to match the resized image
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2) # draw rectangle

cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# image = cv2.imread("./data/images/IMG_2550 conv.jpeg")
# resp = detect_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), device=device)

# # Get the width and height of the image
# height, width, _ = image.shape

# # Resize the image to a smaller size for display
# resized_image = cv2.resize(image, (int(width/2.5), int(height/2.5)))

# # Iterate over each detection and draw on the resized image
# for bbox in resp:
#     xmin, ymin, xmax, ymax = bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']
#     xmin, ymin, xmax, ymax = int(xmin/2.5), int(ymin/2.5), int(xmax/2.5), int(ymax/2.5)  # Scale the coordinates to match the resized image
#     cv2.rectangle(resized_image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)  # Draw rectangle

# # Display the resized image with detections
# cv2.imshow("Resized Image", resized_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()