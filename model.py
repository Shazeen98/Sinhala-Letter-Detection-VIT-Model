import cv2
import torch
import json
import imutils

import numpy as np

from PIL import Image
from imutils import contours
from torchvision import transforms


def union(a, b):
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[0] + a[2], b[0] + b[2]) - x
    h = max(a[1] + a[3], b[1] + b[3]) - y
    return [x, y, w, h]


def _intersect(a, b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0] + a[2], b[0] + b[2]) - x
    h = min(a[1] + a[3], b[1] + b[3]) - y
    if w < 0 or h < 0:
        return False
    return True


def _is_near(a, b):
    if (abs(a[0] - b[0]) < 10 and abs((a[3] + a[1]) - (b[1])) < 10) or (
            abs(a[1] - b[1]) < 10 and abs((a[2] + a[0]) - (b[0])) < 10):
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
            j = i + 1
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
    def __init__(self, model_path="./model/latest.pth", id2label="./model/id2label.json",
                 label2id="./model/label2id.json") -> None:

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path)

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

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Resize((224, 80), antialias=True),
            transforms.Resize((224, 224), antialias=True),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])

    def load_model(self, path):
        # model = resnet18()
        # done = False
        # in_features = model.fc.in_features
        #
        # fc_layers = []
        # while not done:
        #     if (in_features / 32) < 20:
        #         fc_layers.append(torch.nn.Linear(in_features=in_features, out_features=20, bias=True))
        #         print("Finishing up...")
        #         break
        #     else:
        #         out_features = int(in_features / 32)
        #         print(in_features, out_features)
        #         fc_layers.append(torch.nn.Linear(in_features=in_features, out_features=out_features, bias=True))
        #         in_features = out_features
        #
        # model.fc = torch.nn.Sequential(*fc_layers)
        # # model = HandWrittenChar()
        # # model = torch.load(path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        # model.load_state_dict(torch.load(path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))

        return torch.load(path, map_location=self.device)

    def detect(self, image, word):
        if isinstance(image, Image.Image):
            image = np.asarray(image)

        img_gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
        img_gray = 255 - img_gray
        ret, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

        cnts = imutils.grab_contours((contours, hierarchy))
        (cnts, boundingBoxes) = imutils.contours.sort_contours(cnts, method="left-to-right")
        boundingBoxes = _group_rectangles(list(boundingBoxes))

        height, width, _ = image.shape

        with open("predictions/predictions.jsonl", "a+") as preds:

            letters = []
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

                letter = image[y: y + h, x:x + w]
                letter = self._classify(letter)
                letters.append(letter[0])
                pred = {
                    "actual": word[i],
                    "predicted": letter[0],
                    "conf": letter[1]
                }
                preds.write(json.dumps(pred) + "\n")

        return letters

    def _classify(self, image):

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        image = self.transform(image).to(self.device)

        if image.ndim == 3:
            image = image.unsqueeze(dim=0)

        preds = torch.softmax(self.model(image), dim=1)

        if preds.ndim == 2:
            class_ids = torch.argmax(preds, dim=1)
            classes = ""
            confs = []
            for i, class_id in enumerate(class_ids):
                classes = classes + self.id2label[class_id.item()]
                confs.append(torch.max(preds).item())

            return classes, confs

        else:
            class_id = torch.argmax(preds)
            return self.id2label[class_id.item()], [torch.max(preds).item()]
