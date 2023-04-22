import cv2
import torch
import json
import imutils

import numpy as np

from PIL import Image
from torchvision import transforms
from torchvision.models.resnet import resnet50


def union(a,b):
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[0]+a[2], b[0]+b[2]) - x
    h = max(a[1]+a[3], b[1]+b[3]) - y
    return [x, y, w, h]

def _intersect(a,b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0]+a[2], b[0]+b[2]) - x
    h = min(a[1]+a[3], b[1]+b[3]) - y
    if  w < 0 or h < 0:
        return False
    return True

def _is_near(a, b):
    if (abs(a[0] - b[0]) < 10 and abs((a[3] + a[1]) - (b[1])) < 10) or (abs(a[1] - b[1]) < 10 and abs((a[2] + a[0]) - (b[0])) < 10):
        return True
    else:
        return False

def merge(a, b):
    # if (abs(a[0] - b[0]) < 10 and abs((a[3] + a[1]) - (b[1])) < 10):
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = abs(x - max(a[0] + a[2], b[0] + b[2]))
    h = abs(y - max(a[1] + a[3], b[1] + b[3]))

    print([x, y, w, h])

    return [x, y, w, h]


def _group_rectangles(rec):
    """
    Uion intersecting rectangles.
    Args:
        rec - list of rectangles in form [x, y, w, h]
    Return:
        list of grouped ractangles 
    """
    tested = [False for i in range(len(rec))]
    final = []
    i = 0
    while i < len(rec):
        if not tested[i]:
            j = i+1
            while j < len(rec):
                if not tested[j] and _intersect(rec[i], rec[j]):
                    rec[i] = union(rec[i], rec[j])
                    tested[j] = True
                    j = i
                elif not tested[j] and _is_near(rec[i], rec[j]):
                    rec[i] = merge(rec[i], rec[j])
                    tested[j] = True
                    j = i
                j += 1
            final += [rec[i]]
        i += 1

    return final


class LetterDetection():
    def __init__(self, model_path="./model_best_resnet34.pth", id2label="./id2label.json", label2id="./label2id.json") -> None:
        self.model = model_path

        with open(id2label, 'r') as idlabel:
            _id2label = json.load(idlabel)

        self.id2label = {}
        for key, value in _id2label.items():
            self.id2label[int(key)] = value

        with open(label2id, 'r') as labelid:
            _label2id = json.load(labelid)

        self.label2id = {}
        for key, value in _label2id.items():
            self.label2id[key] = int(value)

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def load_model(self, path):
        model = resnet50()
        model.load_state_dict(torch.load(path))
        return model

    def detect(self, image):
        if isinstance(image, Image):
            image = np.asarray(image)
        
        img_gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
        img_gray = 255 - img_gray
        ret, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

        cnts = imutils.grab_contours((contours, hierarchy))
        (cnts, boundingBoxes) = imutils.contours.sort_contours(cnts, method="left-to-right")
        boundingBoxes = _group_rectangles(list(boundingBoxes))

        height, width, _ = image.shape

        word = []
        for i, bbox in enumerate(boundingBoxes):
            x, y, w, h = bbox
            if x - 10 >= 0:
                x = x - 5
            
            if y - 10 >= 0:
                y = y - 5

            if w + 10 < width:
                w = w + 10
            else:
                w = w + 5

            if y + h + 10 < height:
                h = h + 10
            else:
                h = h + 5

            letter = image[y: y+h, x:x+w]
            word.append(letter[0])


    def _classify(self, image):

        if isinstance(image, np.ndarray):
            image = self.preprocess(Image.fromarray(image))

        elif isinstance(image, list):
            image = torch.tensor(image)

        if image.ndim == 3:
            image = image.unsqueeze(dim=0)

        preds = self.model(image)

        if preds.ndim == 2:
            class_ids = torch.argmax(preds, dim=1)
            classes = ""
            for i, class_id in enumerate(class_ids):
                classes = classes + self.id2label[class_id]

            return classes

        else:
            class_id = torch.argmax(preds)
            return {0: self.id2label[class_id]}

        