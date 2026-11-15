"""Evaluation-only precision addendum to the unchanged frozen learning-rate design.

The original recipe_check.py, candidate configurations and selection rule remain byte-identical.
Freeze this addendum before selection/test inference; then use its guarded FP32 launcher.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/recipe_check"
SOURCES = ("scripts/recipe_check_fp32.py", "slurm/evaluate_recipe_check_fp32.sbatch")
AUDITS = ("results/failure_analysis_numerics/audit.json", "results/failure_analysis_numerics/summary.md",
          "results/failure_analysis_numerics/provenance.json")
POLICY = {"matmul_allow_tf32": False, "cudnn_allow_tf32": False}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def load_recipe(repo=REPO):
    spec = importlib.util.spec_from_file_location("frozen_recipe_check", repo / "scripts/recipe_check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_static_amendment(out, repo=REPO):
    path = out / "evaluation_amendment.json"
    amendment = json.loads(path.read_text())
    if amendment.get("schema_version") != 1 or amendment.get("precision_policy") != POLICY:
        raise ValueError("Unknown evaluation precision amendment")
    if amendment.get("original_design_sha256") != sha(out / "design.json"):
        raise ValueError("Original frozen design changed after evaluation amendment")
    expected = {name: sha(repo / name) for name in SOURCES}
    if amendment.get("additional_source_hashes") != expected:
        raise ValueError("Evaluation addendum source changed after amendment freeze")
    if set(amendment.get("numerical_audit_hashes", {})) != set(AUDITS):
        raise ValueError("Amendment must bind the complete numerical audit")
    for name, expected_hash in amendment["numerical_audit_hashes"].items():
        if sha(repo / name) != expected_hash:
            raise ValueError("The numerical evidence bound to the amendment changed")
    proof = amendment.get("pretest_absence_proof", {})
    if proof.get("selection_absent") is not True or proof.get("selected_test_artifacts_absent") is not True:
        raise ValueError("Amendment lacks the required pre-selection/pre-test absence record")
    if (proof.get("verified_utc") != amendment["created_utc"]
            or proof.get("checked_selection_path") != str((out / "selection.json").resolve())
            or proof.get("checked_selected_eval_directory") != str((out / "eval").resolve())):
        raise ValueError("Absence proof must identify this output directory and amendment timestamp")
    if datetime.fromisoformat(amendment["created_utc"]).tzinfo is None:
        raise ValueError("Amendment timestamp must include its UTC timezone")
    return amendment


def check_amendment(out, recipe, repo=REPO):
    # This still checks every original code/config/checkpoint/data hash, unchanged.
    recipe.check_design(out)
    return check_static_amendment(out, repo)


def freeze_amendment(out, recipe, repo=REPO):
    path = out / "evaluation_amendment.json"
    if path.exists():
        return check_amendment(out, recipe, repo)
    recipe.check_design(out)
    if (out / "selection.json").exists() or any((out / "eval").glob("*")):
        raise ValueError("Refuse retrospective amendment: validation selection or selected-test artifacts already exist")
    created = datetime.now(timezone.utc).isoformat()
    amendment = {"schema_version": 1, "scope": "evaluation-only FP32 precision amendment; training and validation selection unchanged",
                 "created_utc": created, "original_design_sha256": sha(out / "design.json"),
                 "precision_policy": POLICY, "additional_source_hashes": {name: sha(repo / name) for name in SOURCES},
                 "numerical_audit_hashes": {name: sha(repo / name) for name in AUDITS},
                 "pretest_absence_proof": {"verified_utc": created, "selection_absent": True,
                    "selected_test_artifacts_absent": True, "checked_selection_path": str((out / "selection.json").resolve()),
                    "checked_selected_eval_directory": str((out / "eval").resolve())},
                 "preserved": ["original design.json and frozen source hashes", "all training/configuration settings",
                    "two candidate learning rates per family", "two seeds and 200000 updates",
                    "validation-only selection criterion and 1e-9 tie rule", "reuse of frozen original evaluations"],
                 "reason": "CPU/CUDA numerical audit found cuDNN TF32 affected near-orthogonal estimates; disable TF32 before new selected-test inference"}
    write_json(path, amendment)
    lines = ["# Evaluation-only FP32 amendment", "", f"Frozen at {created}, before validation selection or selected-test artifacts existed.", "",
             "The original `design.json`, all of its frozen files, training candidates, checkpoints and validation-only "
             "selection rule are preserved. This addendum binds a separate launcher and batch script by SHA-256; "
             "the original training jobs continue to use the original design checker.", "",
             "Before new GPU test inference, the launcher disables CUDA matmul and cuDNN TF32. A numerical audit "
             "on frozen checkpoints found that this reduces a CPU/CUDA SI-SDR discrepancy from 0.029 dB to "
             "0.00056 dB without changing the signal-processing implementation. The audit, extra source files and "
             "original design are bound in `evaluation_amendment.json`.", "",
             "The launcher calls the unchanged validation selector before test inference. New evaluation metadata "
             "records precision flags, amendment and launcher hashes; cached new evaluations must carry matching "
             "precision provenance. Previously frozen original test evaluations retain their own original backend. "
             "This is a numerical-evaluation amendment, not a new training recipe or a change in model selection."]
    (out / "evaluation_amendment.md").write_text("\n".join(lines) + "\n")
    return check_amendment(out, recipe, repo)


def configure_fp32(torch):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.backends.cuda.matmul.allow_tf32 is not False or torch.backends.cudnn.allow_tf32 is not False:
        raise RuntimeError("Failed to disable TF32 before selected-test inference")


def verify_precision_metadata(metadata, amendment_hash, launcher_hash):
    policy = metadata.get("evaluation_precision", {})
    if (metadata.get("evaluation_amendment_sha256") != amendment_hash
            or metadata.get("evaluation_launcher_sha256") != launcher_hash
            or policy.get("matmul_allow_tf32") is not False or policy.get("cudnn_allow_tf32") is not False):
        raise ValueError("Selected evaluation cache lacks matching FP32 amendment provenance")


def check_selection_time(out, amendment):
    selected = json.loads((out / "selection.json").read_text())
    if datetime.fromisoformat(selected["created_utc"]) < datetime.fromisoformat(amendment["created_utc"]):
        raise ValueError("Validation selection predates the evaluation-only amendment")


@contextmanager
def guarded_evaluation(recipe, out, amendment, torch, repo=REPO):
    original_evaluate, original_cache = recipe.evaluate_one, recipe.selected_cache
    amendment_hash = sha(out / "evaluation_amendment.json")
    launcher_hash = sha(repo / SOURCES[0])

    def evaluate_one(*args, **kwargs):
        check_static_amendment(out, repo)
        check_selection_time(out, amendment)
        if torch.backends.cuda.matmul.allow_tf32 is not False or torch.backends.cudnn.allow_tf32 is not False:
            raise RuntimeError("TF32 must be disabled before selected-test inference")
        original_evaluate(*args, **kwargs)
        identity = args[1] if len(args) > 1 else kwargs["identity"]
        path = out / "eval" / f"{identity['config']}_seed{identity['seed']}_test.json"
        metadata = json.loads(path.read_text())
        metadata.update({"evaluation_amendment_sha256": amendment_hash, "evaluation_launcher_sha256": launcher_hash,
                         "evaluation_precision": POLICY.copy(), "evaluation_torch_version": torch.__version__})
        write_json(path, metadata)

    def selected_cache(*args, **kwargs):
        frame = original_cache(*args, **kwargs)
        if frame is not None:
            check_static_amendment(out, repo)
            check_selection_time(out, amendment)
            identity = args[2] if len(args) > 2 else kwargs["identity"]
            path = out / "eval" / f"{identity['config']}_seed{identity['seed']}_test.json"
            verify_precision_metadata(json.loads(path.read_text()), amendment_hash, launcher_hash)
        return frame

    recipe.evaluate_one, recipe.selected_cache = evaluate_one, selected_cache
    try:
        yield
    finally:
        recipe.evaluate_one, recipe.selected_cache = original_evaluate, original_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--freeze-amendment", action="store_true")
    actions.add_argument("--check-amendment", action="store_true")
    actions.add_argument("--evaluate-selected", action="store_true")
    actions.add_argument("--summarize", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    recipe = load_recipe()
    if args.freeze_amendment:
        freeze_amendment(args.out, recipe)
        print("Evaluation-only FP32 amendment frozen before selection/test")
        return
    amendment = check_amendment(args.out, recipe)
    if args.check_amendment:
        print("Original design and separate FP32 evaluation amendment verified")
        return
    import torch
    configure_fp32(torch)
    with guarded_evaluation(recipe, args.out, amendment, torch):
        if args.evaluate_selected:
            recipe.evaluate_selected(args.out, args.device)
        else:
            recipe.summarize(args.out)
    check_selection_time(args.out, amendment)


if __name__ == "__main__":
    main()
