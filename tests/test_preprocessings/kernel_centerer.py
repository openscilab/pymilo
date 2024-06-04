from numpy import array_equal
from sklearn.preprocessing import KernelCenterer
from sklearn.metrics.pairwise import pairwise_kernels
from pymilo.utils.test_pymilo import report_status
from pymilo.transporters.preprocessing_transporter import PreprocessingTransporter
from util import get_path, write_and_read

MODEL_NAME = "KernelCenterer"

def kernel_centerer():
    X = [[ 1., -2., 2.],
         [-2., 1., 3.],
         [ 4., 1., -2.]]
    kernel = pairwise_kernels(X, metric='linear')
    _kernel_centerer = KernelCenterer().fit(kernel)
    pre_result = _kernel_centerer.transform(kernel)

    pt = PreprocessingTransporter()
    post_pymilo_pre_model = pt.deserialize_pre_module(
        write_and_read(
            pt.serialize_pre_module(_kernel_centerer),
            get_path(MODEL_NAME)))
    post_result = post_pymilo_pre_model.transform(kernel)

    comparison_result = array_equal(pre_result, post_result)
    report_status(comparison_result, MODEL_NAME)
    assert comparison_result
