from numpy import array_equal, random
from sklearn.datasets import load_sample_images
from sklearn.feature_extraction import image
from pymilo.transporters.feature_extraction_transporter import FeatureExtractorTransporter
from pymilo.utils.test_pymilo import report_status
from util import get_path, write_and_read

MODEL_NAME = "PatchExtractor"

def patch_extractor():
    X = load_sample_images().images[1]
    X = X[None, ...]
    print(f"Image shape: {X.shape}")
    pe = image.PatchExtractor(patch_size=(10, 10), random_state=random.RandomState(42))
    pre_result = pe.transform(X)

    fe = FeatureExtractorTransporter()

    print("before: \n",pe.__dict__)
    post_pymilo_pre_model = fe.deserialize_fe_module(
        write_and_read(
            fe.serialize_fe_module(pe),
            get_path(MODEL_NAME)))
    print("before: \n",post_pymilo_pre_model.__dict__)
    post_result = post_pymilo_pre_model.transform(X)
    comparison_result = array_equal(pre_result, post_result)
    report_status(comparison_result, MODEL_NAME)
    assert comparison_result
