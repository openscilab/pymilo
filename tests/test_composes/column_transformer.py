from numpy import array, array_equal
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Normalizer, MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from util import get_path, write_and_read
from pymilo.transporters.compose_transporter import ComposeTransporter
from pymilo.utils.test_pymilo import report_status

MODEL_NAME = "ColumnTransformer"

def column_transformer():
    model = ColumnTransformer(
        [("norm1", Normalizer(norm='l1'), [0, 1]),
        ("norm2", Normalizer(norm='l1'), slice(2, 4))])
    X = array([[0., 1., 2., 2.],
                [1., 1., 0., 1.]])
    # Normalizer scales each row of X to unit norm. A separate scaling
    # is applied for the two first and two last elements of each
    # row independently.
    pre_result = model.fit_transform(X)

    ct = ComposeTransporter()
    post_pymilo_ct_model = ct.deserialize_compose_internal_model(
        write_and_read(
            ct.serialize_compose_internal_model(model),
            get_path(MODEL_NAME,1)))
    post_result = post_pymilo_ct_model.transform(X)

    comparison_result = array_equal(pre_result, post_result)
    report_status(comparison_result, MODEL_NAME)
    assert comparison_result


def complex_column_transformer():
    # Use a plain 2D numpy array (no pandas dependency). With numpy inputs, select columns by index.
    X = array(
        [["First item", 3.0], ["second one here", 4.0], ["Is this the last?", 5.0]],
        dtype=object,
    )
    ct = ColumnTransformer(
        [("text_preprocess", CountVectorizer(), 0),
        ("num_preprocess", MinMaxScaler(), [1])])
    pre_result = ct.fit_transform(X)

    pt = ComposeTransporter()
    post_pymilo_pre_model = pt.deserialize_compose_internal_model(
        write_and_read(
            pt.serialize_compose_internal_model(ct),
            get_path(MODEL_NAME,2)))
    post_result = post_pymilo_pre_model.transform(X)

    comparison_result = array_equal(pre_result, post_result)
    report_status(comparison_result, MODEL_NAME)
    assert comparison_result


def nested_column_transformer_with_pipeline():
    # Numeric-only example to keep output dense and easily comparable
    X = array([[0., 1., 2., 3.],
               [1., 1., 0., 1.],
               [2., 0., 1., 0.]])

    # Inner ColumnTransformer working on array indices (post StandardScaler)
    inner_ct = ColumnTransformer([
        ("minmax_first_two", MinMaxScaler(), [0, 1])
    ])

    # Pipeline used as a transformer inside the outer ColumnTransformer
    from sklearn.pipeline import Pipeline
    pipe = Pipeline([
        ("scale", MinMaxScaler()),
        ("inner_ct", inner_ct),
    ])

    # Outer ColumnTransformer applies the pipeline on all columns
    outer_ct = ColumnTransformer([
        ("pipe", pipe, slice(0, 4))
    ])

    pre_result = outer_ct.fit_transform(X)

    pt = ComposeTransporter()
    post_model = pt.deserialize_compose_internal_model(
        write_and_read(
            pt.serialize_compose_internal_model(outer_ct),
            get_path(MODEL_NAME, 3)))
    post_result = post_model.transform(X)

    comparison_result = array_equal(pre_result, post_result)
    report_status(comparison_result, MODEL_NAME)
    assert comparison_result
