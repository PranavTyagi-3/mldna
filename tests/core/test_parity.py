import pickle

import numpy as np
import pytest
from onnx import TensorProto, helper
from sklearn.linear_model import LinearRegression

from mlops.core.parity import (
    ParityError,
    check_parity,
    infer_schema,
    register_format_runner,
)


def _onnx_linear_model(coef, intercept):
    coef = coef.astype(np.float32).reshape(-1, 1)
    intercept = np.array([intercept], dtype=np.float32)

    x_info = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, coef.shape[0]])
    y_info = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, 1])
    coef_init = helper.make_tensor("coef", TensorProto.FLOAT, coef.shape, coef.flatten())
    intercept_init = helper.make_tensor("intercept", TensorProto.FLOAT, intercept.shape, intercept)

    matmul = helper.make_node("MatMul", ["X", "coef"], ["matmul_out"])
    add = helper.make_node("Add", ["matmul_out", "intercept"], ["Y"])

    graph = helper.make_graph(
        [matmul, add], "linear", [x_info], [y_info], [coef_init, intercept_init]
    )
    # ir_version pinned to 10: onnx's default (13) is newer than what our
    # pinned onnxruntime==1.20.1 supports (see DECISIONS.md D#23).
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)


def _make_bundle(tmp_path, coefficients):
    tmp_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(0)
    X = rng.rand(50, 4).astype(np.float32)
    y = X @ np.array(coefficients, dtype=np.float32) + 3.0
    model = LinearRegression().fit(X, y)

    pkl_path = tmp_path / "model.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(model, f)

    # Hand-built rather than via skl2onnx: skl2onnx's dynamic_class_creation
    # segfaults on Windows when imported alongside this project's full test
    # suite (see DECISIONS.md D#23). A hand-built graph exercises the exact
    # same MatMul+Add a linear model's conversion would produce, without the
    # fragile dependency.
    onnx_model = _onnx_linear_model(model.coef_, model.intercept_)
    onnx_dir = tmp_path / "onnx"
    onnx_dir.mkdir()
    (onnx_dir / "model.onnx").write_bytes(onnx_model.SerializeToString())

    sample_path = tmp_path / "holdout_sample.npz"
    np.savez(sample_path, X=X[:10])

    return pkl_path, onnx_dir, sample_path


def test_matching_pkl_and_onnx_passes(tmp_path):
    pkl_path, onnx_dir, sample_path = _make_bundle(tmp_path, [1.0, 2.0, -1.0, 0.5])

    result = check_parity(pkl_path, [onnx_dir], sample_path, atol=1e-4)

    assert result.formats["onnx"] < 1e-4


def test_mismatched_conversion_raises(tmp_path):
    pkl_path, _, sample_path = _make_bundle(tmp_path, [1.0, 2.0, -1.0, 0.5])
    _, other_onnx_dir, _ = _make_bundle(tmp_path / "other", [5.0, -3.0, 2.0, 1.0])

    with pytest.raises(ParityError, match="Parity check failed"):
        check_parity(pkl_path, [other_onnx_dir], sample_path, atol=1e-4)


def test_missing_sample_key_raises(tmp_path):
    pkl_path, onnx_dir, _ = _make_bundle(tmp_path, [1.0, 2.0, -1.0, 0.5])
    bad_sample = tmp_path / "bad_sample.npz"
    np.savez(bad_sample, wrong_key=np.zeros((5, 4)))

    with pytest.raises(ParityError, match="must contain an array named 'X'"):
        check_parity(pkl_path, [onnx_dir], bad_sample)


def test_unregistered_format_raises(tmp_path):
    pkl_path, _, sample_path = _make_bundle(tmp_path, [1.0, 2.0, -1.0, 0.5])
    fake_dir = tmp_path / "openvino"
    fake_dir.mkdir()

    with pytest.raises(ParityError, match="No format runner registered for 'openvino'"):
        check_parity(pkl_path, [fake_dir], sample_path)


def test_custom_format_runner_is_used(tmp_path):
    pkl_path, _, sample_path = _make_bundle(tmp_path, [1.0, 2.0, -1.0, 0.5])
    fake_dir = tmp_path / "fakefmt"
    fake_dir.mkdir()

    calls = []

    def fake_runner(format_dir, sample):
        calls.append(format_dir)
        model = pickle.load(open(pkl_path, "rb"))
        return model.predict(sample)

    register_format_runner("fakefmt", fake_runner)

    result = check_parity(pkl_path, [fake_dir], sample_path)

    assert calls == [fake_dir]
    assert result.formats["fakefmt"] == 0.0


def test_infer_schema(tmp_path):
    pkl_path, onnx_dir, sample_path = _make_bundle(tmp_path, [1.0, 2.0, -1.0, 0.5])
    result = check_parity(pkl_path, [onnx_dir], sample_path, atol=1e-4)

    schema = infer_schema(result)

    assert schema["inputs"][0]["shape"] == [None, 4]
    assert schema["outputs"][0]["dtype"].startswith("float")
