from binarizer import binarizer

PREPROCESSINGS = [binarizer]

def test_full():
    for pre in PREPROCESSINGS:
        pre()
