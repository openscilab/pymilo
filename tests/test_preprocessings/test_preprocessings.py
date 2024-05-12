from binarizer import binarizer
from one_hot_encoder import one_hot_encoder

PREPROCESSINGS = [binarizer, one_hot_encoder]

def test_full():
    for pre in PREPROCESSINGS:
        pre()
