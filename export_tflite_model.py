"""
Export the trained sklearn DNN Pipeline to TFLite for on-device inference.

Usage:
    python export_tflite_model.py

Reads:
    /Volumes/External/projects/models/classification/dnn.pkl
        - sklearn Pipeline: StandardScaler → MLPClassifier(256, 128, 64)
        - Trained on 13 BASE_FEATURES

Writes:
    loanly_app/assets/models/loan_approval.tflite   (TFLite flatbuffer)
    loanly_app/assets/models/loan_approval_norm.json (scaler mean/std + feature meta)
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_PKL = Path("/Volumes/External/projects/models/classification/dnn.pkl")
FLUTTER_ASSETS = Path(__file__).resolve().parent.parent / "projects" / "AS Projects" / "loanly_app" / "assets" / "models"
# Fallback: place next to this script if the Flutter path doesn't exist
FALLBACK_ASSETS = Path(__file__).resolve().parent / "exported_tflite"

def _resolve_assets_dir() -> Path:
    candidate = Path("/Volumes/External/projects/AS Projects/loanly_app/assets/models")
    if candidate.parent.parent.exists():
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate
    fallback = FALLBACK_ASSETS
    fallback.mkdir(parents=True, exist_ok=True)
    print(f"[warn] Flutter assets dir not found, writing to {fallback}")
    return fallback

# ---------------------------------------------------------------------------
# BASE_FEATURES — must match pipeline_utils.py exactly and Dart feature order
# ---------------------------------------------------------------------------
BASE_FEATURES = [
    "Age",
    "Monthly_Income",
    "Credit_Score",
    "Loan_Amount_Requested",
    "Existing_Debts",
    "Num_Previous_Loans",
    "Payment_History_Encoded",
    "Employment_Status_Encoded",
    "Debt_to_Income_Ratio",
    "Loan_to_Income_Ratio",
    "Repayment_Consistency",
    "Application_Velocity",
    "Credit_Utilization_Score",
]


def _load_pipeline(path: Path):
    print(f"Loading pipeline from {path} ...")
    pipe = joblib.load(path)
    scaler = pipe.named_steps["scaler"]
    mlp = pipe.named_steps["model"]
    print(f"  Scaler mean shape : {scaler.mean_.shape}")
    print(f"  MLP layers        : {[c.shape for c in mlp.coefs_]}")
    return scaler, mlp


def _build_keras_model(scaler, mlp):
    """Rebuild sklearn MLPClassifier as a tf.keras.Sequential and copy weights."""
    try:
        import tensorflow as tf
    except ImportError:
        sys.exit("TensorFlow not installed. Run: pip install tensorflow")

    n_features = len(BASE_FEATURES)
    model = tf.keras.Sequential(name="loan_dnn")
    model.add(tf.keras.layers.InputLayer(shape=(n_features,), name="input"))

    activation_map = {"relu": "relu", "tanh": "tanh", "logistic": "sigmoid"}
    hidden_activation = activation_map.get(mlp.activation, "relu")
    output_activation = activation_map.get(mlp.out_activation_, "sigmoid")

    layer_names = ["dense_256", "dense_128", "dense_64", "output"]
    for i, (W, b) in enumerate(zip(mlp.coefs_, mlp.intercepts_)):
        is_last = i == len(mlp.coefs_) - 1
        act = output_activation if is_last else hidden_activation
        name = layer_names[i] if i < len(layer_names) else f"dense_{i}"
        layer = tf.keras.layers.Dense(
            W.shape[1], activation=act, name=name,
            kernel_initializer="zeros", bias_initializer="zeros",
        )
        # Build by calling with dummy input to create weights
        dummy = tf.zeros((1, W.shape[0]))
        layer(dummy)
        layer.set_weights([W, b])
        model.add(layer)

    # Quick sanity: shapes
    model.summary()
    return model


def _export_tflite(keras_model, out_path: Path):
    import tensorflow as tf

    # TF 2.16+ serialises FULLY_CONNECTED as op-version 12, which older TFLite
    # runtimes (bundled by tflite_flutter < 0.11.0) cannot load.
    # Fix A (preferred): upgrade tflite_flutter to ^0.11.0 in pubspec.yaml.
    # Fix B (fallback):  use tensorflow==2.15.* for this export step.
    tf_ver = tuple(int(x) for x in tf.__version__.split(".")[:2])
    if tf_ver >= (2, 16):
        print(
            f"[warn] TensorFlow {tf.__version__} generates FULLY_CONNECTED op-v12.\n"
            "       Upgrade tflite_flutter to ^0.11.0 in pubspec.yaml, or re-export\n"
            "       with tensorflow==2.15.* to stay compatible with older runtimes."
        )

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    # Keep float32 only; avoid dynamic-range quantization which can push newer op paths.
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.target_spec.supported_types = [tf.float32]
    tflite_model = converter.convert()
    out_path.write_bytes(tflite_model)
    size_kb = len(tflite_model) / 1024
    print(f"TFLite model written → {out_path}  ({size_kb:.1f} KB)")
    return tflite_model


def _save_norm_json(scaler, out_path: Path):
    norm = {
        "features": BASE_FEATURES,
        "mean": scaler.mean_.tolist(),
        "std": scaler.scale_.tolist(),
        "payment_history_map": {"poor": 0, "fair": 1, "good": 2, "excellent": 3},
        "employment_status_map": {"unemployed": 0, "employed": 1, "self_employed": 2},
    }
    out_path.write_text(json.dumps(norm, indent=2))
    print(f"Norm JSON written  → {out_path}")
    return norm


def _sanity_check(pipe, tflite_bytes, norm):
    """Compare sklearn pipeline output vs TFLite on a few synthetic rows."""
    import tensorflow as tf

    rng = np.random.default_rng(0)
    # Build 5 plausible rows in BASE_FEATURES order
    n = 5
    age = rng.uniform(22, 65, n)
    income = rng.uniform(1000, 8000, n)
    credit = rng.uniform(300, 850, n)
    loan = rng.uniform(500, 20000, n)
    debts = rng.uniform(0, income * 0.6)
    prev = rng.integers(0, 8, n).astype(float)
    ph = rng.integers(0, 4, n).astype(float)
    emp = rng.integers(0, 3, n).astype(float)
    dti = debts / income
    lti = loan / income
    rep_con = ph / 3.0
    app_vel = prev / 10.0
    cred_util = debts / (credit * 0.1 + 1e-9)

    X = np.column_stack([age, income, credit, loan, debts, prev, ph, emp, dti, lti, rep_con, app_vel, cred_util]).astype(np.float32)

    # sklearn pipeline prediction
    sk_probs = pipe.predict_proba(X)[:, 1]

    # TFLite prediction
    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    tflite_probs = []
    for row in X:
        interpreter.set_tensor(inp["index"], row.reshape(1, -1))
        interpreter.invoke()
        tflite_probs.append(float(interpreter.get_tensor(out["index"])[0][0]))

    print("\nSanity check — sklearn vs TFLite probability of default:")
    print(f"  {'sklearn':>10}  {'tflite':>10}  {'|diff|':>10}")
    for sk, tfl in zip(sk_probs, tflite_probs):
        print(f"  {sk:10.6f}  {tfl:10.6f}  {abs(sk-tfl):10.6f}")
    max_diff = max(abs(sk - tfl) for sk, tfl in zip(sk_probs, tflite_probs))
    print(f"\n  Max absolute difference: {max_diff:.6f}")
    if max_diff > 0.05:
        print("  [warn] Difference > 0.05 — check quantization or weight transfer")
    else:
        print("  [ok] Outputs match within tolerance")


def main():
    if not MODEL_PKL.exists():
        sys.exit(f"Model not found: {MODEL_PKL}")

    assets_dir = _resolve_assets_dir()
    tflite_path = assets_dir / "loan_approval.tflite"
    norm_path = assets_dir / "loan_approval_norm.json"

    scaler, mlp = _load_pipeline(MODEL_PKL)
    keras_model = _build_keras_model(scaler, mlp)
    tflite_bytes = _export_tflite(keras_model, tflite_path)
    norm = _save_norm_json(scaler, norm_path)

    # Reload full pipeline for sanity check
    pipe = joblib.load(MODEL_PKL)
    _sanity_check(pipe, tflite_bytes, norm)

    print("\nDone. Asset files:")
    print(f"  {tflite_path}")
    print(f"  {norm_path}")


if __name__ == "__main__":
    main()
