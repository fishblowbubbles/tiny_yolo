import matplotlib.pyplot as plt
import cv2
import numpy as np
import time
import torch


def timeit(f):
    def timer(*args, **kwargs):
        start = time.time()
        output = f(*args, **kwargs)
        elapsed = time.time() - start
        print("Execution Time of '{}': {:.2f} seconds"
              .format(f.__name__, elapsed))
        return output
    return timer


def read_img(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (416, 416))
    return img


def show_img(path):
    img = read_img(path)
    img = np.array(img)
    plt.imshow(img)

    
def load_img(path):
    img = read_img(path)
    img = img.transpose((2, 0, 1))
    img = img[np.newaxis, ...] / 255.0
    img = torch.from_numpy(img).float()
    return img


def load_names(path):
    names = []
    with open(path, "r") as file:
        lines = file.read()
        lines = lines.strip()
        lines = lines.split("\n")
        for line in lines:
            names.append(line)
    return names


def xywh2xyxy(x):
    """
    Converts (x (center), y(center), width, height) to (x (top left), y (top, left), x (bottom right), y (bottom right)).
    """
    output = x.new(x.shape)
    output[..., 0] = x[..., 0] - x[..., 2] / 2
    output[..., 1] = x[..., 1] - x[..., 3] / 2
    output[..., 2] = x[..., 0] + x[..., 2] / 2
    output[..., 3] = x[..., 1] + x[..., 3] / 2
    return output


def area(rect):
    x1, y1, x2, y2 = rect[:, 0], rect[:, 1], rect[:, 2], rect[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    return area


def iou(a, b):
    """
    Calculate Intersection Over Union (IOU) of two bounding boxes.
    """
    x1, y1 = torch.max(a[..., 0], b[..., 0]), torch.max(a[..., 1], b[..., 1])
    x2, y2 = torch.min(a[..., 2], b[..., 2]), torch.min(a[..., 3], b[..., 3])
    intersection = torch.clamp(x2 - x1 + 1, min=0) * torch.clamp(y2 - y1 + 1, min=0) 
    union = area(a) + area(b) - intersection
    iou = intersection / union
    return iou


def filter_nm(detections, overlap_threshold):
    coordinates, confidence, indices = (detections[:, :4], 
                                        detections[:, 4], 
                                        detections[:, 6])
    # locate detections with overlapping bounding boxes and the same class prediction
    overlap = iou(detections[0, :4].unsqueeze(dim=0), coordinates) >= overlap_threshold
    match = detections[0, 6] == indices
    to_remove = overlap & match
    # adjust final bounding box size according to probabilities of non-maximum bounding boxes
    rescale_by = detections[to_remove, 4].unsqueeze(dim=1)
    detections[0, :4] = (rescale_by * detections[to_remove, :4]).sum(dim=0) / rescale_by.sum()
    # save final detection
    bbox = detections[0]
    # remove non-maximum detections
    detections = detections[~to_remove]
    return bbox, detections


def nms(x, confidence_threshold, overlap_threshold):
    """
    Perform non-maximum suppression over all output bounding boxes.
    """
    x[..., :4] = xywh2xyxy(x[..., :4])
    output = [None for _ in range(len(x))]
    for i, detections in enumerate(x):
        # remove detections below confidence threshold
        detections = detections[detections[:, 4] >= confidence_threshold]
        if not detections.shape[0] > 0: continue
        # remove non-maximum class predictions
        coordinates, confidence, class_predictions = (detections[:, :4], 
                                                      detections[:, 4:5], 
                                                      detections[:, 5:])
        probabilities, indices = class_predictions.max(dim=1, keepdim=True)
        # sort by detection score in descending order
        scores = probabilities * confidence
        detections = detections[probabilities.argsort(descending=True)]
        # reconstruct tensor
        detections = torch.cat((coordinates, confidence, probabilities.float(), indices.float()), dim=1)
        # extract and save maximum detections (adjusted bounding boxes)
        bboxes = []
        while detections.shape[0] > 0:
            bbox, detections = filter_nm(detections, overlap_threshold)
            bboxes.append(bbox)
        output[i] = bboxes
    return output