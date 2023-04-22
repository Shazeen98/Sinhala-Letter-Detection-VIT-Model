import random

from PIL import Image
from flask import Flask, request
from flask_cors import CORS

from model import LetterDetection
from db import init_db, Word

app = Flask(__name__)
app.config.from_object('config.Config')

CORS(app)
init_db(app)

letter_detection = LetterDetection("./model/model_best.pth", "./model/id2letter.json", "./model/letter2id.json")

@app.route("/suggest_word", methods=['POST'])
def get_word():
    data = request.form
    
    if "image" in request.files:
        image = Image.open(request.files.getlist()[0])
        word = list(data["word"])

        detected = letter_detection.detect(image)
        suggested = []
        for letter_orig, letter_detect in zip(word, detected):
            if not (letter_orig == letter_detect):
                words = list(Word.objects(word__contains=letter_detect))
                freqs = []
                for word in words:
                    freqs.append(word.freq[letter_detect])

                suggested.append(words[freqs.index(max(freqs))].word)

        if len(suggested) >= 1:
            return random.choice(suggested)
        else:
            return suggested[0]

    else:
        return random.choice(list(Word.objects(word__contains="අ"))).word
    

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)


        
