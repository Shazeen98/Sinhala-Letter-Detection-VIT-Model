import re
import json
import random
import torch

from PIL import Image
from flask import Flask, request
from flask_cors import CORS

from model import LetterDetection
from db import init_db, Word

from transformers import ViTModel, ViTConfig
from torchvision.models.resnet import resnet18

# from ultralytics import YOLO

app = Flask(__name__)
app.config.from_object('config.Config')

CORS(app)
init_db(app)


class ViT(torch.nn.Module):

    def __init__(self, config=ViTConfig(), num_labels=20,
                 model_checkpoint='google/vit-base-patch16-224-in21k'):
        super(ViT, self).__init__()

        self.vit = ViTModel.from_pretrained(model_checkpoint, add_pooling_layer=False)
        self.classifier = (
            torch.nn.Linear(config.hidden_size, num_labels)
        )

    def forward(self, x):
        x = self.vit(x)['last_hidden_state']
        # Use the embedding of [CLS] token
        output = self.classifier(x[:, 0, :])

        return output


letter_detection = LetterDetection(id2label="./model/id2letter.json", label2id="./model/letter2id.json")


@app.route("/suggest_word", methods=['POST'])
def get_word():
    data = request.form

    if "image" in request.files:
        word_orig = re.findall(r'\w\W?', data["word"])
        images = request.files.getlist("image")
        suggested = []

        with open("predictions/predictions.jsonl", "a+") as preds:
            for i, (letter_orig, image) in enumerate(zip(word_orig, images)):
                image = Image.open(image).convert("RGB").resize((80, 80))

                # detected = letter_detection.detect(image, word)
                detected, conf = letter_detection._classify(image)

                pred = {
                    "actual": letter_orig,
                    "predicted": detected,
                    "conf": conf
                }

                preds.write(json.dumps(pred) + "\n")

                if not (letter_orig == detected):
                    words = list(Word.objects(word__contains=letter_orig))
                    freqs = []
                    for word in words:
                        freqs.append(word.freq[letter_orig])

                    suggested.append(words[freqs.index(max(freqs))].word)

            if len(suggested):
                return re.findall(r'\w\W?', random.choice(suggested))
            else:
                return re.findall(r'\w\W?', random.choice(list(Word.objects(word__contains="අ"))).word)

    else:

        return re.findall(r'\w\W?', random.choice(list(Word.objects(word__contains="අ"))).word)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
