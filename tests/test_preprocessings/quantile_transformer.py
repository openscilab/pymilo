from numpy import array_equal, random, sort
from sklearn.preprocessing import QuantileTransformer
from pymilo.utils.test_pymilo import report_status
from pymilo.transporters.preprocessing_transporter import PreprocessingTransporter
from util import get_path, write_and_read

MODEL_NAME = "QuantileTransformer"

def quantile_transformer():
    rng = random.RandomState(0)
    X = sort(rng.normal(loc=0.5, scale=0.25, size=(25, 1)), axis=0)

    _quantile_transformer = QuantileTransformer(n_quantiles=10, random_state=0).fit(X)
    pre_result = _quantile_transformer.transform(X)

    pt = PreprocessingTransporter()
    post_pymilo_pre_model = pt.deserialize_pre_module(
        write_and_read(
            pt.serialize_pre_module(_quantile_transformer),
            get_path(MODEL_NAME)))
    post_result = post_pymilo_pre_model.transform(X)

    comparison_result = array_equal(pre_result, post_result)
    report_status(comparison_result, MODEL_NAME)
    assert comparison_result
