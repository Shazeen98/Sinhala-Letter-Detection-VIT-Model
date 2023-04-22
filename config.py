import os
from datetime import timedelta


basedir = os.path.abspath(os.path.dirname(__file__))


class ConfigMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(ConfigMeta, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Config(metaclass=ConfigMeta):
    FLASK_DEBUG = True
    MONGODB_SETTINGS = {
        "db": "Words",
        "host": "mongodb+srv://mhmmdshazeen:CCjDMaI3LBWz4Yz0@cluster0.jmm07qw.mongodb.net/Words?retryWrites=true&w=majority"
    }
