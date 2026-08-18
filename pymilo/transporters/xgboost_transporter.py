# -*- coding: utf-8 -*-
"""PyMilo transporter for XGBoost Booster objects and related fields."""
import json
import math

from .transporter import AbstractTransporter
from ..utils.util import check_str_in_iterable
from ..pymilo_param import (
    xgboost_support,
    XGBoostBooster,
    XGBOOST_NOT_INSTALLED,
)

PYMILO_XGBOOST_BOOSTER = "pymilo-xgboost-booster"
PYMILO_XGBOOST_CALLABLE = "pymilo-xgboost-callable"
PYMILO_XGBOOST_CALLBACKS = "pymilo-xgboost-callbacks"

GPU_DEVICE_PREFIXES = ("cuda", "gpu")
GPU_TREE_METHODS = {"gpu_hist", "gpu_exact"}
CPU_DEVICE = "cpu"
CPU_TREE_METHOD = "hist"


def is_xgboost_booster(obj):
    """
    Check whether the given object is an XGBoost Booster.

    :param obj: given object
    :type obj: any
    :return: check result as bool
    """
    if not xgboost_support or XGBoostBooster is None:
        return False
    return isinstance(obj, XGBoostBooster)


def is_gpu_device(device):
    """
    Check whether the given device specifier refers to a GPU.

    :param device: device name such as ``cpu``, ``cuda``, ``cuda:0`` or ``gpu``
    :type device: str or None
    :return: check result as bool
    """
    if device is None:
        return False
    device_name = str(device).strip().lower()
    return any(device_name == prefix or device_name.startswith(prefix + ":") for prefix in GPU_DEVICE_PREFIXES)


def is_xgboost_gpu_available():
    """
    Check whether the installed XGBoost build reports CUDA support.

    This does not allocate a GPU context and does not run a GPU kernel.

    :return: check result as bool
    """
    if not xgboost_support:
        return False
    try:
        import xgboost
        if hasattr(xgboost, "build_info"):
            info = xgboost.build_info()
            return bool(info.get("USE_CUDA"))
    except Exception:
        return False
    return False


def rewrite_gpu_fields_to_cpu(payload):
    """
    Recursively rewrite GPU device / tree-method fields in a nested payload to CPU.

    The original payload is modified in place and also returned.

    :param payload: nested dict / list structure (XGBoost JSON model or config)
    :type payload: any
    :return: the same payload with GPU fields rewritten
    """
    if isinstance(payload, dict):
        device = payload.get("device")
        if is_gpu_device(device):
            payload["device"] = CPU_DEVICE
        tree_method = payload.get("tree_method")
        if isinstance(tree_method, str) and tree_method in GPU_TREE_METHODS:
            payload["tree_method"] = CPU_TREE_METHOD
        for value in payload.values():
            rewrite_gpu_fields_to_cpu(value)
    elif isinstance(payload, list):
        for item in payload:
            rewrite_gpu_fields_to_cpu(item)
    return payload


def extract_requested_device(model_json, config, fallback=None):
    """
    Extract the device requested by an XGBoost model payload.

    :param model_json: parsed XGBoost ``save_raw(raw_format="json")`` body
    :type model_json: dict or None
    :param config: parsed XGBoost ``save_config()`` body
    :type config: dict or None
    :param fallback: value used when no device field is present
    :type fallback: str or None
    :return: device name as str or None
    """
    candidates = []
    if isinstance(config, dict):
        generic = config.get("learner", {}).get("generic_param", {})
        if isinstance(generic, dict):
            candidates.append(generic.get("device"))
    if isinstance(model_json, dict):
        generic = model_json.get("learner", {}).get("generic_param", {})
        if isinstance(generic, dict):
            candidates.append(generic.get("device"))
    for item in candidates:
        if item is not None:
            return item
    return fallback


def booster_to_transparent_dict(booster):
    """
    Convert an XGBoost Booster to a fully transparent JSON-compatible dictionary.

    The official XGBoost JSON model (trees, weights, splits, learner params) is stored
    as parsed JSON rather than a binary / UBJSON buffer.

    :param booster: fitted XGBoost Booster
    :type booster: xgboost.core.Booster
    :return: transparent dictionary representation
    """
    if not xgboost_support:
        raise ImportError(XGBOOST_NOT_INSTALLED)

    import xgboost
    raw = _save_booster_json_bytes(booster)
    model_json = json.loads(raw.decode("utf-8"))
    try:
        config = json.loads(booster.save_config())
    except Exception:
        config = None

    attributes = {}
    try:
        attributes = booster.attributes() or {}
    except Exception:
        attributes = {}

    feature_names = None
    feature_types = None
    try:
        feature_names = booster.feature_names
    except Exception:
        feature_names = None
    try:
        feature_types = booster.feature_types
    except Exception:
        feature_types = None

    trees_dump = None
    try:
        dumped = booster.get_dump(dump_format="json")
        trees_dump = [json.loads(tree) for tree in dumped]
    except Exception:
        trees_dump = None

    requested_device = extract_requested_device(model_json, config)

    payload = {
        "xgboost_version": getattr(xgboost, "__version__", None),
        "model-json": model_json,
        "config": config,
        "feature_names": feature_names,
        "feature_types": feature_types,
        "attributes": attributes,
        "requested_device": requested_device,
        "trees-dump": trees_dump,
        "num_boosted_rounds": _safe_call(booster.num_boosted_rounds),
        "num_features": _safe_call(booster.num_features),
        "best_iteration": _safe_getattr(booster, "best_iteration"),
        "best_score": _safe_getattr(booster, "best_score"),
    }
    return payload


def booster_from_transparent_dict(payload, map_gpu_to_cpu=True):
    """
    Rebuild an XGBoost Booster from a transparent dictionary produced by PyMilo.

    When ``map_gpu_to_cpu`` is True and the stored device is a GPU device, the payload
    is rewritten to CPU before ``load_model`` so deserialization works without a GPU.

    :param payload: previously serialized booster dictionary
    :type payload: dict
    :param map_gpu_to_cpu: rewrite GPU device fields to CPU when GPU is unavailable
        or when the caller requests a CPU-safe load
    :type map_gpu_to_cpu: bool
    :return: reconstructed ``xgboost.core.Booster``
    """
    if not xgboost_support or XGBoostBooster is None:
        raise ImportError(XGBOOST_NOT_INSTALLED)
    if not isinstance(payload, dict):
        raise TypeError("XGBoost booster payload must be a dictionary.")

    model_json = payload.get("model-json")
    if model_json is None:
        raise ValueError("XGBoost booster payload is missing the transparent `model-json` field.")

    load_body = json.loads(json.dumps(model_json))
    config = payload.get("config")
    requested_device = payload.get("requested_device")
    if requested_device is None:
        requested_device = extract_requested_device(load_body, config)

    should_map = map_gpu_to_cpu and (
        is_gpu_device(requested_device) or not is_xgboost_gpu_available()
    )
    if is_gpu_device(requested_device) and should_map:
        rewrite_gpu_fields_to_cpu(load_body)
        if isinstance(config, dict):
            rewrite_gpu_fields_to_cpu(config)

    raw = json.dumps(load_body).encode("utf-8")
    booster = XGBoostBooster()
    booster.load_model(bytearray(raw))

    if is_gpu_device(requested_device) and should_map:
        _restore_optional_field(lambda: booster.set_param({"device": CPU_DEVICE}))

    feature_names = payload.get("feature_names")
    if feature_names is not None:
        _restore_optional_field(lambda: setattr(booster, "feature_names", list(feature_names)))
    feature_types = payload.get("feature_types")
    if feature_types is not None:
        _restore_optional_field(lambda: setattr(booster, "feature_types", list(feature_types)))

    attributes = payload.get("attributes") or {}
    for key, value in attributes.items():
        attr_value = None if value is None else str(value)
        _restore_optional_field(lambda k=key, v=attr_value: booster.set_attr(**{str(k): v}))
    return booster


def wrap_booster_payload(booster):
    """
    Wrap a Booster as a pymilo-bypass dictionary used inside model ``__dict__``.

    :param booster: fitted XGBoost Booster
    :type booster: xgboost.core.Booster
    :return: dictionary with ``pymilo-bypass`` and the transparent booster body
    """
    return {
        "pymilo-bypass": True,
        PYMILO_XGBOOST_BOOSTER: booster_to_transparent_dict(booster),
    }


def unwrap_booster_payload(content, map_gpu_to_cpu=True):
    """
    Unwrap a pymilo booster dictionary back to a Booster.

    :param content: serialized content
    :type content: dict
    :param map_gpu_to_cpu: rewrite GPU device fields to CPU when needed
    :type map_gpu_to_cpu: bool
    :return: reconstructed Booster or the original content
    """
    if check_str_in_iterable(PYMILO_XGBOOST_BOOSTER, content):
        return booster_from_transparent_dict(content[PYMILO_XGBOOST_BOOSTER], map_gpu_to_cpu=map_gpu_to_cpu)
    return content


def _save_booster_json_bytes(booster):
    """
    Export a Booster as UTF-8 JSON bytes using the official XGBoost JSON format.

    :param booster: fitted XGBoost Booster
    :type booster: xgboost.core.Booster
    :return: JSON bytes
    """
    try:
        raw = booster.save_raw(raw_format="json")
        return bytes(raw)
    except TypeError:
        # Older XGBoost builds accept no raw_format argument.
        raw = booster.save_raw()
        if _looks_like_json(raw):
            return bytes(raw)
        raise ValueError(
            "This XGBoost build cannot export a Booster as JSON. "
            "Upgrade XGBoost to a version that supports save_raw(raw_format='json')."
        )


def _looks_like_json(raw):
    """
    Check whether a raw buffer starts with a JSON object / array.

    :param raw: raw bytes or bytearray
    :type raw: bytes or bytearray
    :return: check result as bool
    """
    if not raw:
        return False
    first = bytes(raw).lstrip()[:1]
    return first in (b"{", b"[")


def _restore_optional_field(action):
    """
    Run an optional Booster restore action and report whether it succeeded.

    Feature names, feature types and device are auxiliary; a failure must not
    block reconstruction of the learned trees.

    :param action: zero-argument callable performing the restore
    :type action: callable
    :return: True when the action succeeded, otherwise False
    """
    try:
        action()
        return True
    except Exception:
        return False


def _safe_call(func):
    """
    Call ``func`` and return None when it raises.

    :param func: zero-argument callable
    :type func: callable
    :return: func result or None
    """
    try:
        return func()
    except Exception:
        return None


def _safe_getattr(obj, name):
    """
    Return ``getattr(obj, name)`` or None when it raises.

    :param obj: given object
    :type obj: any
    :param name: attribute name
    :type name: str
    :return: attribute value or None
    """
    try:
        return getattr(obj, name)
    except Exception:
        return None


def _is_nan_number(value):
    """
    Check whether the given value is a NaN floating number.

    :param value: given value
    :type value: any
    :return: check result as bool
    """
    if isinstance(value, float):
        return math.isnan(value)
    try:
        import numpy as np
        return isinstance(value, np.floating) and np.isnan(value)
    except Exception:
        return False


class XGBoostTransporter(AbstractTransporter):
    """Customized PyMilo Transporter developed to handle XGBoost Booster objects and related fields."""

    def serialize(self, data, key, model_type):
        """
        Serialize XGBoost Booster instances and a few non-JSON XGBoost fields.

        serialize the data[key] of the given model which type is model_type.
        basically in order to fully serialize a model, we should traverse over all the keys of its data dictionary and
        pass it through the chain of associated transporters to get fully serialized.

        :param data: the internal data dictionary of the given model
        :type data: dict
        :param key: the special key of the data param, which we're going to serialize its value(data[key])
        :type key: object
        :param model_type: the model type of the ML model, which data dictionary is given as the data param
        :type model_type: str
        :return: pymilo serialized output of data[key]
        """
        value = data[key]
        if is_xgboost_booster(value):
            data[key] = wrap_booster_payload(value)
            return data[key]
        if key == "missing" and _is_nan_number(value):
            data[key] = {
                "np-type": "numpy.nan",
                "value": "NaN"
            }
            return data[key]
        if key == "callbacks" and value is not None:
            data[key] = {
                "pymilo-bypass": True,
                PYMILO_XGBOOST_CALLBACKS: _serialize_callbacks(value),
            }
            return data[key]
        if key == "eval_metric" and callable(value):
            data[key] = {
                "pymilo-bypass": True,
                PYMILO_XGBOOST_CALLABLE: getattr(value, "__name__", None),
            }
            return data[key]
        return value

    def deserialize(self, data, key, model_type):
        """
        Deserialize previously serialized XGBoost Booster instances and related fields.

        deserialize the data[key] of the given model which type is model_type.
        basically in order to fully deserialize a model, we should traverse over all the keys of its serialized data dictionary and
        pass it through the chain of associated transporters to get fully deserialized.

        :param data: the internal data dictionary of the associated JSON file of the ML model generated by pymilo export.
        :type data: dict
        :param key: the special key of the data param, which we're going to deserialize its value(data[key])
        :type key: object
        :param model_type: the model type of the ML model
        :type model_type: str
        :return: pymilo deserialized output of data[key]
        """
        content = data[key]
        if check_str_in_iterable(PYMILO_XGBOOST_BOOSTER, content):
            return unwrap_booster_payload(content, map_gpu_to_cpu=True)
        if check_str_in_iterable(PYMILO_XGBOOST_CALLBACKS, content):
            # Custom callback objects cannot be reconstructed transparently.
            return None
        if check_str_in_iterable(PYMILO_XGBOOST_CALLABLE, content):
            return content.get(PYMILO_XGBOOST_CALLABLE)
        return content


def _serialize_callbacks(callbacks):
    """
    Serialize XGBoost callback objects as their type names.

    Custom callback instances are not reconstructed; only a transparent name list is kept.

    :param callbacks: callback list or single callback
    :type callbacks: list or object
    :return: list of callback type names
    """
    if callbacks is None:
        return None
    if not isinstance(callbacks, (list, tuple)):
        callbacks = [callbacks]
    names = []
    for item in callbacks:
        names.append(getattr(item, "__class__", type(item)).__name__)
    return names
