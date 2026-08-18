# XGBoost Support in PyMilo — Evaluation Report

Date: 2026-08-18  
Environment: Linux, Python 3.12.3, scikit-learn 1.6.1, XGBoost 3.4.1, numpy 2.2.4  
PyMilo version left at **1.6** (changes listed under `[Unreleased]`).

## Goal

Add support for all XGBoost models to PyMilo, aligned with the existing chain + transporter architecture, with **transparent JSON** output (not a binary/UBJSON blob), plus tests, Bandit, pydocstyle, and coverage.

## What was implemented

XGBoost is an **optional** dependency (`pip install pymilo[xgboost]` or a bare `pip install xgboost`). If it is missing, the table entries become `NOT_SUPPORTED` and Import/Export raise a clear `ImportError`.

Supported models (sklearn API + core booster):

| Model | Role |
| --- | --- |
| `XGBClassifier` | binary + multiclass |
| `XGBRegressor` | regression, including `booster="dart"` and `booster="gblinear"` |
| `XGBRanker` | learning-to-rank |
| `XGBRFClassifier` | RF-style classifier wrapper (deprecated upstream, still exported) |
| `XGBRFRegressor` | RF-style regressor wrapper |
| `XGBModel` | generic sklearn base |
| `Booster` | low-level `xgboost.core.Booster` |

Dask/Spark XGBoost estimators were **not** added. They require extra runtimes and are not serializable as standalone local estimators the same way.

### Transparency (not binary)

Export still uses PyMilo’s existing JSON envelope:

```json
{
  "data": { ... },
  "sklearn_version": "...",
  "pymilo_version": "1.6",
  "model_type": "XGBClassifier",
  "xgboost_version": "3.4.1"
}
```

The fitted booster is **not** stored as `save_raw()` UBJSON/bytes. It is parsed from the official XGBoost JSON format (`save_raw(raw_format="json")`) and written as nested JSON. A real export from this environment contained:

- `data._Booster.pymilo-xgboost-booster.model-json.learner.gradient_booster.model.trees[]`
- per-tree fields: `split_conditions`, `split_indices`, `left_children`, `right_children`, `base_weights`, `loss_changes`, `sum_hessian`, `tree_param`
- learner config (`save_config()`), feature names/types, attributes
- `trees-dump` (human-readable `get_dump(dump_format="json")`)
- constructor params (`n_estimators`, `max_depth`, `device`, `objective`, …)
- fitted extras such as `n_classes_` and `evals_result_`

`Import` accepts the same three existing channels as the rest of PyMilo: file path, `json_dump` string, and URL.

GPU-trained payloads are still valid JSON. On import, `device=cuda|gpu` is rewritten to CPU when a GPU is not available (or when the caller requests a CPU-safe load). The original `requested_device` stays in the JSON for inspection.

## Files changed and why

### New

| File | Why |
| --- | --- |
| `pymilo/transporters/xgboost_transporter.py` | Serializes/deserializes `Booster`, `missing=NaN`, callbacks, callable `eval_metric`; GPU→CPU rewrite helpers |
| `pymilo/chains/xgboost_chain.py` | Chain for wrappers + standalone `Booster` |
| `tests/test_xgboosts/*` | Model, transparency, mocked-network, simulated-GPU, routing, and coverage tests |
| `XGBOOST_SUPPORT_REPORT.md` | This report |

### Modified

| File | Why |
| --- | --- |
| `pymilo/pymilo_param.py` | Optional XGBoost imports, `XGBOOST_MODEL_TABLE`, `EXPORTED_MODELS_PATH["XGBOOST"]`, `ALL_SUPPORTED_CATEGORIES`, version/install messages |
| `pymilo/chains/util.py` | Route `"XGBOOST"` / XGBoost objects through the new chain (including inner Pipeline models) |
| `pymilo/pymilo_func.py` | `get_xgboost_version()`; print all categories including XGBoost |
| `pymilo/pymilo_obj.py` | Write/read `xgboost_version`; warn on mismatch |
| `pymilo/utils/util.py` | `get_sklearn_class()` searches `ALL_SUPPORTED_CATEGORIES` |
| `pymilo/utils/data_exporter.py` | `prepare_simple_ranking_datasets()` |
| `pymilo/utils/test_pymilo.py` | `pymilo_prediction_test()` for ranker/booster/multiclass |
| `pymilo/transporters/preprocessing_transporter.py` | `serialize_spline()` now exports public `t,c,k` fields (see below) |
| `setup.py` | Extra: `'xgboost': ['xgboost>=1.6.0']` |
| `README.md` | XGBoost row in supported-models table |
| `SUPPORTED_MODELS.md` | XGBoost section |
| `CHANGELOG.md` | `[Unreleased]` notes |
| `tests/test_exceptions/import_exceptions.py` | URL tests now mock the network (no live HTTP) |

`coverage_core.xml` was generated during test runs and **not** kept in the tree.

## Configs / defaults

| Item | Default / choice |
| --- | --- |
| XGBoost install | Optional. Core `requirements.txt` unchanged |
| Extra | `pip install pymilo[xgboost]` → `xgboost>=1.6.0` |
| CPU device | Tests always pass `device="cpu"`, `n_jobs=1`, `verbosity=0`, and set `CUDA_VISIBLE_DEVICES=""` |
| Import GPU models | Map `cuda`/`gpu`/`gpu_hist` → `cpu`/`hist` when GPU is unavailable |
| Export format | Official XGBoost JSON (`raw_format="json"`), pretty-printed by `Export.to_json()` |
| Version metadata | `sklearn_version` kept; `xgboost_version` added only for XGBoost models |
| `XGBRF*` | Still supported; XGBoost 3.4 emits a `FutureWarning` (upstream deprecation) |

## Tests added

Under `tests/test_xgboosts/`:

| File | What it covers |
| --- | --- |
| `test_xgboosts.py` | Runner for all model families (creates `tests/exported_xgboosts/`) |
| `xgb_classifier.py` | Binary `XGBClassifier` via `pymilo_classification_test` |
| `xgb_classifier_multiclass.py` | 3-class + `predict_proba` |
| `xgb_regressor.py` | `XGBRegressor` |
| `xgb_ranker.py` | `XGBRanker` with qid |
| `xgb_rf_classifier.py` / `xgb_rf_regressor.py` | RF wrappers |
| `xgb_model.py` | Generic `XGBModel` |
| `xgb_booster.py` | Standalone `Booster` + `DMatrix` predict |
| `xgb_dart.py` / `xgb_gblinear.py` | Other booster types |
| `xgb_early_stopping.py` | `evals_result_` / `best_iteration` |
| `xgb_feature_names.py` | Feature-name round-trip |
| `xgb_pipeline.py` | `Pipeline(StandardScaler, XGBClassifier)` inner-model JSON |
| `test_xgboost_transparency.py` | JSON is structured (trees, not bytes); file / `json_dump` / batch / unfitted |
| `test_xgboost_network.py` | `Import(url=...)` with **mocked** `requests` / `download_model` |
| `test_xgboost_gpu.py` | Device helpers + simulated CUDA payload remapped to CPU |
| `test_xgboost_exceptions.py` | Wrong chain, bad payload |
| `test_xgboost_routing.py` | `get_transporter`, CLI class lookup, version warning |
| `test_xgboost_more_coverage.py` | Guards, helpers, inner deserialize |
| `xgboost_test_helpers.py` | CPU kwargs (unique name so it does not clash with other `util.py` test modules) |

Existing URL exception tests (`valid_url_valid_file`, etc.) now load the local `linear_regression.json` through a mock instead of hitting GitHub.

## Exact commands and real outputs

### XGBoost suite (CPU only)

```bash
python -m pytest tests/test_xgboosts -v --tb=line
```

Real result:

```
================== 36 passed, 1 skipped, 4 warnings in 3.26s ===================
```

The single skip is `test_real_gpu_training_disabled` (unconditional skip; see GPU section).

All 36 passing names are in `/tmp/pytest_xgboost.txt` from this session. Highlights:

```
test_xgboost_gpu.py::test_simulated_gpu_payload_falls_back_to_cpu_on_import PASSED
test_xgboost_network.py::test_import_from_url_with_mocked_network PASSED
test_xgboost_network.py::test_import_from_url_via_requests_session_mock PASSED
test_xgboost_network.py::test_import_from_url_network_failure_is_simulated PASSED
test_xgboost_transparency.py::test_to_json_is_human_readable_and_not_binary PASSED
test_xgboosts.py::test_full PASSED
```

`test_full` trains/exports/imports: classifier (binary+multi), regressor, ranker, both RF wrappers, `XGBModel`, `Booster`, DART, gblinear, early stopping, feature names, and a Pipeline.

### Full core suite (CI equivalent, streaming ignored)

```bash
python -m pytest . --ignore=./tests/test_ml_streaming --cov=pymilo --cov-report=term --cov-report=term-missing --tb=line
```

Real result:

```
collected 51 items
...
tests/test_xgboosts/test_xgboosts.py .                                   [100%]
...
TOTAL                                                        2948   1031   1050     96    66%
============ 50 passed, 1 skipped, 38 warnings in 77.81s (0:01:17) =============
```

Exit code **0**.

New-module coverage from that run:

| Module | Cover |
| --- | --- |
| `pymilo/chains/xgboost_chain.py` | **100%** |
| `pymilo/transporters/xgboost_transporter.py` | **85%** |
| `pymilo/utils/data_exporter.py` | **100%** |
| `pymilo/chains/util.py` | **100%** |
| `pymilo/transporters/preprocessing_transporter.py` | **96%** (was lower before the spline JSON fix) |

Overall 66% includes `pymilo/streaming/*` at 0% because streaming tests were ignored (same as CI’s core job). This does **not** reduce existing sklearn-chain coverage.

### Bandit

```bash
python -m bandit -r pymilo -s B311
```

```
Test results:
No issues identified.

Code scanned:
Total lines of code: 6713
...
Total issues (by severity):
Undefined: 0
Low: 0
Medium: 0
High: 0
```

### Pydocstyle

```bash
python -m pydocstyle -v
```

Checked every package module including `xgboost_chain.py` and `xgboost_transporter.py`. Exit code **0** (no findings).

### Vulture

```bash
python -m vulture pymilo/ otherfiles/ setup.py --min-confidence 65 --exclude=__init__.py --sort-by-size
```

Exit code **0** (no unused-code findings).

### Version check

```bash
python otherfiles/version_check.py
```

```
Version tag tests passed!
Passed : 7/7
```

## Network

No test in this work opens a real socket to the internet.

- New XGBoost URL tests patch `pymilo.pymilo_obj.download_model` and `pymilo.utils.util.requests.Session`.
- Existing `valid_url_valid_file` / `valid_url_invalid_file` / `invalid_url` now use the same mock style and the local `tests/test_exceptions/valid_jsons/linear_regression.json`.

`tests/test_ml_streaming` was **not** executed (CI runs it as a separate job; it starts local REST/WebSocket servers). That is not XGBoost work.

## GPU — what exists, what was simulated, what was not run

There were **no** GPU tests in the repo before this change (`gpu`/`cuda` grep was empty).

Added `tests/test_xgboosts/test_xgboost_gpu.py`:

| Test | Behavior |
| --- | --- |
| `test_gpu_helpers_recognize_cuda_specifiers` | `cuda`, `cuda:0`, `gpu` vs `cpu` |
| `test_rewrite_gpu_fields_to_cpu_is_recursive` | Nested dict rewrite |
| `test_exported_cpu_model_records_cpu_device_metadata` | Exported JSON records `device=cpu` |
| `test_simulated_gpu_payload_falls_back_to_cpu_on_import` | CPU-trained JSON is rewritten to claim `device=cuda`, then Import remaps to CPU; predictions match |
| `test_gpu_availability_probe_does_not_allocate_a_device` | Reads `xgboost.build_info()` only |
| `test_real_gpu_training_disabled` | **SKIPPED always** — documents a real-CUDA entry point |

**Not tested on GPU (cannot be tested on CPU):**

- Actual `XGBClassifier(device="cuda").fit(...)`
- `tree_method="gpu_hist"` training
- CUDA predictor / inplace GPU predict
- Multi-GPU `cuda:1` execution
- Numerical identity between a GPU-trained booster and its CPU-imported copy when both devices exist

All training and prediction in this session used **CPU**. `CUDA_VISIBLE_DEVICES` is forced empty in the GPU test module and helper.

## Extra compatibility fix (not XGBoost-specific)

`SplineTransformer` tests failed on this SciPy/sklearn pair because `BSpline.__dict__` now holds a private `_BSpline` plus module objects, which `json.dumps` cannot encode. `serialize_spline()` now writes the public constructor fields (`t`, `c`, `k`, `extrapolate`, `axis`). That unblocked `tests/test_preprocessings` and raised `preprocessing_transporter.py` coverage to 96%.

## How to use

```python
from xgboost import XGBClassifier
from pymilo import Export, Import

model = XGBClassifier(n_estimators=50, device="cpu").fit(X, y)
Export(model).save("xgb.json")          # inspectable JSON, including trees
restored = Import("xgb.json").to_model()
restored.predict(X)
```

`Export(model).to_json()` returns the same payload as a string. `Import(json_dump=...)` and `Import(url=...)` work as for sklearn models.
