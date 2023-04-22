from mongoengine import DictField, StringField
from flask_mongoengine import MongoEngine

db = MongoEngine()

class Word(db.Document):
    word = StringField()
    freq = DictField()


# def connect_db():
#     DB_URI = "mongodb+srv://mhmmdshazeen:CCjDMaI3LBWz4Yz0@cluster0.jmm07qw.mongodb.net/Words?retryWrites=true&w=majority"
#     db = connect(db="Words", host=DB_URI)
#     return db

def init_db(app):
    db.init_app(app)
