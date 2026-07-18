import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort


class ParityError(Exception):
    pass


FORMAT_RUNNERS = {}


def register_format_runner(name, fn):
    FORMAT_RUNNERS[name] = fn


def _run_onnx(format_dir, sample):
    onnx_files = list(Path(format_dir).glob("*.onnx"))
    if not onnx_files:
        raise ParityError(f"No .onnx file found in {format_dir}")

    session = ort.InferenceSession(str(onnx_files[0]))
    input_name = session.get_inputs()[0].name
    return session.run(None, {input_name: sample})[0]


register_format_runner("onnx", _run_onnx)


def load_sample(sample_path):
    data = np.load(sample_path)
    if "X" not in data:
        raise ParityError(
            f"{sample_path} must contain an array named 'X' (the held-out "
            f"parity-check sample)."
        )
    return data["X"]


def run_pkl(pkl_path, sample):
    with open(pkl_path, "rb") as f:
        model = pickle.load(f)
    if not hasattr(model, "predict"):
        raise ParityError(
            f"{pkl_path} has no .predict() method. The parity check assumes "
            f"the pkl exposes model.predict(X), the convention shared by "
            f"sklearn/xgboost/lightgbm/catboost."
        )
    return np.asarray(model.predict(sample))


@dataclass
class ParityResult:
    sample_path: str
    atol: float
    formats: dict
    sample: np.ndarray
    reference: np.ndarray


def check_parity(pkl_path, format_dirs, sample_path, atol=1e-5):
    sample = load_sample(sample_path)
    reference = run_pkl(pkl_path, sample)
    reference_flat = reference.reshape(-1)

    formats = {}
    for format_dir in format_dirs:
        format_dir = Path(format_dir)
        name = format_dir.name
        runner = FORMAT_RUNNERS.get(name)
        if runner is None:
            raise ParityError(
                f"No format runner registered for '{name}'. Register one: "
                f"mlops.core.parity.register_format_runner('{name}', your_fn)"
            )

        predictions = np.asarray(runner(format_dir, sample)).reshape(-1)
        if predictions.shape != reference_flat.shape:
            raise ParityError(
                f"Format '{name}' produced {predictions.shape[0]} output values, "
                f"pkl produced {reference_flat.shape[0]}. Shapes must match."
            )

        max_abs_diff = float(np.max(np.abs(reference_flat - predictions)))
        if max_abs_diff > atol:
            raise ParityError(
                f"Parity check failed for format '{name}': "
                f"max_abs_diff={max_abs_diff} > atol={atol}. The converted "
                f"model disagrees with the pkl it was converted from."
            )
        formats[name] = max_abs_diff

    return ParityResult(
        sample_path=str(sample_path), atol=atol, formats=formats,
        sample=sample, reference=reference,
    )


def infer_schema(result: ParityResult):
    def _field(name, arr):
        shape = [None, *arr.shape[1:]] if arr.ndim > 1 else [None]
        return {"name": name, "dtype": str(arr.dtype), "shape": shape}

    return {
        "inputs": [_field("input", result.sample)],
        "outputs": [_field("output", result.reference)],
    }
