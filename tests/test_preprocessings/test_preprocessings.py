from binarizer import binarizer
from one_hot_encoder import one_hot_encoder
from label_binarizer import label_binarizer

PREPROCESSINGS = [binarizer, one_hot_encoder, label_binarizer]

def test_full():
    for pre in PREPROCESSINGS:
        pre()
