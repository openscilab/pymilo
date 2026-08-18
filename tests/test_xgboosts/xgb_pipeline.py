import json
from numpy import allclose
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from pymilo import Export, Import
from pymilo.utils.data_exporter import prepare_simple_classification_datasets
from xgboost_test_helpers import cpu_estimator_kwargs, xgboost_available

MODEL_NAME = "PipelineXGBClassifier"


def xgb_pipeline():
    if not xgboost_available:
        print("Model: " + MODEL_NAME + " is not supported in this python version.")
        return
    from xgboost import XGBClassifier
    x_train, y_train, x_test, _ = prepare_simple_classification_datasets()
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(**cpu_estimator_kwargs(XGBClassifier))),
        ]
    )
    pipeline.fit(x_train, y_train)
    dumped = Export(pipeline).to_json()
    payload = json.loads(dumped)
    assert payload["model_type"] == "Pipeline"
    dumped_text = dumped
    assert "pymilo-xgboost-booster" in dumped_text
    assert "model-json" in dumped_text
    imported = Import(json_dump=dumped).to_model()
    assert allclose(imported.predict_proba(x_test), pipeline.predict_proba(x_test), rtol=1e-5, atol=1e-6)
    print("Pymilo Test for Model: " + MODEL_NAME + " succeed.")
