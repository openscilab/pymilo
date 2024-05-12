from one_hot_encoder import one_hot_encoder
from label_binarizer import label_binarizer
from label_encoder import label_encoder
from standard_scaler import standard_scaler
from binarizer import binarizer
from function_transformer import function_transformer

PREPROCESSINGS = [one_hot_encoder,
                  label_binarizer,
                  label_encoder,
                  standard_scaler,
                  binarizer,
                  function_transformer]

def test_full():
    for pre in PREPROCESSINGS:
        pre()
