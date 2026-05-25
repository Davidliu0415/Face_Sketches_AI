from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import torch


def git_revision(project_root: Path) -> dict[str, str]:
    def run_git(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *args],
                cwd=project_root,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except Exception:
            return ""

    return {
        "commit": run_git("rev-parse", "HEAD"),
        "short_commit": run_git("rev-parse", "--short", "HEAD"),
        "status_porcelain": run_git("status", "--short"),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def params_snapshot(params_module, num_identities: int | None = None) -> dict[str, Any]:
    snapshot = {
        "embedding_size": params_module.embedding_size,
        "dropout": params_module.dropout,
        "face_num_classes": num_identities or params_module.face_num_classes,
        "sketch_num_classes": num_identities or params_module.sketch_num_classes,
        "arcface_scale": params_module.arcface_scale,
        "arcface_margin": params_module.arcface_margin,
        "triplet_margin": params_module.triplet_margin,
        "triplet_weight": params_module.triplet_weight,
        "batch_hard_weight": getattr(params_module, "batch_hard_weight", 0.0),
        "face_ce_weight": params_module.face_ce_weight,
        "sketch_ce_weight": params_module.sketch_ce_weight,
        "image_size": params_module.image_size,
        "use_amp": params_module.use_amp,
    }
    if num_identities is not None:
        snapshot["num_identities"] = num_identities
    return snapshot


def load_checkpoint_compatible(path: str | Path, model, device) -> dict[str, list[str]]:
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model_state = model.state_dict()
    compatible = {}
    skipped = []

    for key, value in state_dict.items():
        if key in model_state and tuple(model_state[key].shape) == tuple(value.shape):
            compatible[key] = value
        else:
            skipped.append(key)

    missing, unexpected = model.load_state_dict(compatible, strict=False)
    return {
        "loaded": sorted(compatible),
        "skipped_shape_mismatch": sorted(skipped),
        "missing": sorted(missing),
        "unexpected": sorted(unexpected),
    }


def diagnostic_rows(run_id: str, dataset: str, diagnostics: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in diagnostics.get("top_repeated_predictions", []):
        rows.append(
            {
                "run_id": run_id,
                "dataset": dataset,
                "sketch_index": item["sketch_index"],
                "count": item["count"],
                "share": f"{item['share']:.6f}",
                "sketch_path": item["sketch_path"],
            }
        )
    return rows
