"""Launch external graph-condensation baselines with full-node ratios.

The external repositories use different names and ratio conventions.  This
module keeps our experiment contract stable: schedules are expressed as
condensed nodes divided by all original nodes, while each baseline receives the
train/labeled-node ratio expected by its implementation.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


FULL_RATIO_SCHEDULES: dict[str, list[float]] = {
    "ogbn-arxiv": [0.0005, 0.001, 0.0025, 0.005, 0.01],
    "reddit": [0.0005, 0.001, 0.002, 0.005, 0.01],
    "ogbn-products": [0.0005, 0.0025, 0.005, 0.01],
    "ogbn-products-low": [0.0002, 0.0004, 0.0008],
}


@dataclass(frozen=True)
class DatasetStats:
    num_nodes: int
    train_nodes: int
    num_classes: int | None = None


DATASET_STATS: dict[str, DatasetStats] = {
    "ogbn-arxiv": DatasetStats(num_nodes=169_343, train_nodes=90_941, num_classes=40),
    "reddit": DatasetStats(num_nodes=232_965, train_nodes=153_431, num_classes=41),
    "ogbn-products": DatasetStats(num_nodes=2_449_029, train_nodes=196_615, num_classes=47),
}


BASELINES = ("GECC", "DeepCGC", "TGCC", "WbGC", "ClustGDD")

BASELINE_DATASET_SUPPORT: dict[str, set[str]] = {
    "GECC": {"ogbn-arxiv", "reddit", "ogbn-products"},
    "DeepCGC": {"ogbn-arxiv", "reddit"},
    "TGCC": {"ogbn-arxiv", "reddit"},
    "WbGC": {"ogbn-arxiv", "reddit"},
    "ClustGDD": {"ogbn-arxiv", "reddit"},
}

_FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

_OOM_MARKERS: tuple[tuple[str, str], ...] = (
    ("cuda out of memory", "CUDA out of memory"),
    ("cublas_status_alloc_failed", "CUBLAS_STATUS_ALLOC_FAILED"),
    ("cudnn_status_alloc_failed", "CUDNN_STATUS_ALLOC_FAILED"),
    ("outofmemoryerror", "OutOfMemoryError"),
    ("out of memory", "out of memory"),
    ("cannot allocate memory", "cannot allocate memory"),
    ("allocation failed", "allocation failed"),
)


@dataclass
class RunPlan:
    baseline: str
    dataset: str
    schedule_dataset: str
    full_ratio: float
    baseline_ratio: float
    baseline_ratio_denominator: str
    planned_condensed_nodes: int
    seed: int
    gpu: int
    data_root: Path
    baseline_root: Path
    output_root: Path
    run_dir: Path
    cwd: Path
    command: list[str]
    env: dict[str, str]
    stdout_path: Path
    stderr_path: Path
    metrics_path: Path
    summary_path: Path
    status: str = "planned"
    failure_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def canonical_dataset(dataset: str) -> str:
    if dataset == "ogbn-products-low":
        return "ogbn-products"
    return dataset


def ratio_label(ratio: float) -> str:
    text = f"{ratio:.8f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def path_arg(path: Path | str) -> str:
    return Path(path).as_posix()


def absolute_path_arg(path: Path | str) -> str:
    return Path(path).expanduser().resolve().as_posix()


def _path_arg_with_trailing_slash(path: Path | str) -> str:
    text = path_arg(path)
    if not text.endswith("/"):
        text += "/"
    return text


def _fmt_ratio(value: float) -> str:
    return f"{value:.10g}"


def baseline_code_ratio(dataset: str, full_ratio: float, denominator: str = "train") -> float:
    dataset = canonical_dataset(dataset)
    if dataset not in DATASET_STATS:
        raise KeyError(f"Unknown dataset for ratio conversion: {dataset}")
    if denominator == "full":
        return full_ratio
    if denominator != "train":
        raise ValueError(f"Unsupported baseline ratio denominator: {denominator}")
    stats = DATASET_STATS[dataset]
    return (stats.num_nodes * full_ratio) / stats.train_nodes


def planned_condensed_nodes(dataset: str, full_ratio: float) -> int:
    dataset = canonical_dataset(dataset)
    if dataset not in DATASET_STATS:
        raise KeyError(f"Unknown dataset for budget conversion: {dataset}")
    return max(1, int(round(DATASET_STATS[dataset].num_nodes * full_ratio)))


def normalize_baseline(baseline: str) -> str:
    lookup = {name.lower(): name for name in BASELINES}
    key = baseline.lower()
    if key not in lookup:
        raise ValueError(f"Unknown baseline '{baseline}'. Expected one of: {', '.join(BASELINES)}")
    return lookup[key]


def baseline_supports_dataset(baseline: str, dataset: str) -> bool:
    baseline = normalize_baseline(baseline)
    return canonical_dataset(dataset) in BASELINE_DATASET_SUPPORT[baseline]


def _base_run_dir(
    baseline: str,
    schedule_dataset: str,
    full_ratio: float,
    seed: int,
    output_root: Path,
) -> Path:
    return (
        output_root
        / baseline
        / schedule_dataset
        / f"full_ratio_{ratio_label(full_ratio)}"
        / f"seed_{seed}"
    )


def _plan_paths(run_dir: Path) -> tuple[Path, Path, Path, Path]:
    return (
        run_dir / "stdout.log",
        run_dir / "stderr.log",
        run_dir / "metrics.json",
        run_dir / "summary.json",
    )


def _common_env(gpu: int, data_root: Path, run_dir: Path) -> dict[str, str]:
    return {
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "SHADOW_HGC_DATA_ROOT": path_arg(data_root),
        "SHADOW_HGC_RUN_DIR": path_arg(run_dir),
    }


def _unsupported_plan(
    baseline: str,
    dataset: str,
    full_ratio: float,
    seed: int,
    gpu: int,
    data_root: Path,
    baseline_root: Path,
    output_root: Path,
    reason: str,
) -> RunPlan:
    run_dir = _base_run_dir(baseline, dataset, full_ratio, seed, output_root)
    stdout_path, stderr_path, metrics_path, summary_path = _plan_paths(run_dir)
    canonical = canonical_dataset(dataset)
    return RunPlan(
        baseline=baseline,
        dataset=canonical,
        schedule_dataset=dataset,
        full_ratio=full_ratio,
        baseline_ratio=baseline_code_ratio(canonical, full_ratio),
        baseline_ratio_denominator="train",
        planned_condensed_nodes=planned_condensed_nodes(canonical, full_ratio),
        seed=seed,
        gpu=gpu,
        data_root=data_root,
        baseline_root=baseline_root,
        output_root=output_root,
        run_dir=run_dir,
        cwd=baseline_root / baseline,
        command=[],
        env=_common_env(gpu, data_root, run_dir),
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        metrics_path=metrics_path,
        summary_path=summary_path,
        status="unsupported",
        failure_reason=reason,
    )


def build_run_plan(
    baseline: str,
    dataset: str,
    full_ratio: float,
    seed: int,
    gpu: int,
    data_root: Path,
    baseline_root: Path,
    output_root: Path,
    python_executable: str = "python",
) -> RunPlan:
    baseline = normalize_baseline(baseline)
    schedule_dataset = dataset
    dataset = canonical_dataset(dataset)
    baseline_root = Path(baseline_root)
    data_root = Path(data_root)
    output_root = Path(output_root)

    if dataset not in DATASET_STATS:
        raise KeyError(f"Unknown dataset '{schedule_dataset}'")
    if not baseline_supports_dataset(baseline, dataset):
        return _unsupported_plan(
            baseline=baseline,
            dataset=schedule_dataset,
            full_ratio=full_ratio,
            seed=seed,
            gpu=gpu,
            data_root=data_root,
            baseline_root=baseline_root,
            output_root=output_root,
            reason=f"{baseline} launcher does not support {dataset} in the cloned code",
        )

    baseline_ratio = baseline_code_ratio(dataset, full_ratio, denominator="train")
    planned_nodes = planned_condensed_nodes(dataset, full_ratio)
    run_dir = _base_run_dir(baseline, schedule_dataset, full_ratio, seed, output_root)
    stdout_path, stderr_path, metrics_path, summary_path = _plan_paths(run_dir)
    env = _common_env(gpu, data_root, run_dir)

    command, cwd, metadata = _build_baseline_command(
        baseline=baseline,
        dataset=dataset,
        baseline_ratio=baseline_ratio,
        seed=seed,
        data_root=data_root,
        baseline_root=baseline_root,
        run_dir=run_dir,
        python_executable=python_executable,
    )

    return RunPlan(
        baseline=baseline,
        dataset=dataset,
        schedule_dataset=schedule_dataset,
        full_ratio=full_ratio,
        baseline_ratio=baseline_ratio,
        baseline_ratio_denominator="train",
        planned_condensed_nodes=planned_nodes,
        seed=seed,
        gpu=gpu,
        data_root=data_root,
        baseline_root=baseline_root,
        output_root=output_root,
        run_dir=run_dir,
        cwd=cwd,
        command=command,
        env=env,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        metrics_path=metrics_path,
        summary_path=summary_path,
        metadata=metadata,
    )


def _build_baseline_command(
    baseline: str,
    dataset: str,
    baseline_ratio: float,
    seed: int,
    data_root: Path,
    baseline_root: Path,
    run_dir: Path,
    python_executable: str,
) -> tuple[list[str], Path, dict[str, Any]]:
    ratio_text = _fmt_ratio(baseline_ratio)
    run_dir_abs = run_dir.expanduser().resolve()
    data_root_abs = data_root.expanduser().resolve()
    if baseline == "DeepCGC":
        repo = baseline_root / "DeepCGC"
        dataset_arg = "arxiv" if dataset == "ogbn-arxiv" else dataset
        config_folder = run_dir_abs / "deepcgc_config"
        command = [
            python_executable,
            "main.py",
            "--dataset_name",
            dataset_arg,
            "--ratio",
            ratio_text,
            "--raw_data_dir",
            _path_arg_with_trailing_slash(data_root_abs),
            "--result_path",
            path_arg(run_dir_abs / "results") + "/",
            "--cond_folder",
            path_arg(run_dir_abs / "cond_graph") + "/",
            "--config_folder",
            path_arg(config_folder) + "/",
            "--gpu",
            "0",
            "--seed",
            str(seed),
        ]
        return command, repo, {"dataset_arg": dataset_arg, "config_folder": path_arg(config_folder)}

    if baseline == "GECC":
        repo = baseline_root / "GECC" / "graphslim"
        command = [
            python_executable,
            "train_all.py",
            "--dataset",
            dataset,
            "--method",
            "gecc",
            "--gpu_id",
            "0",
            "--reduction_rate",
            ratio_text,
            "--seed",
            str(seed),
            "--save_path",
            path_arg(run_dir_abs / "checkpoints"),
            "--load_path",
            path_arg(data_root_abs),
        ]
        command.extend(_gecc_hyperparams(dataset))
        return command, repo, {}

    if baseline == "TGCC":
        repo = baseline_root / "TGCC"
        section = f"shadow_hgc_{dataset}_{ratio_label(baseline_ratio)}_seed{seed}"
        config_path = run_dir_abs / "tgcc_config.json"
        command = [
            python_executable,
            "tgcc_train.py",
            "--config",
            path_arg(config_path),
            "--section",
            section,
            "--gpu",
            "0",
            "--seed",
            str(seed),
            "--save_dir",
            path_arg(run_dir_abs / "saved_ours"),
            "--log_dir",
            path_arg(run_dir_abs / "logs"),
        ]
        return command, repo, {"config_path": path_arg(config_path), "config_section": section}

    if baseline == "WbGC":
        repo = baseline_root / "WbGC"
        script = "train_gcond_induct_recons.py" if dataset == "reddit" else "train_gcond_transduct_recons.py"
        command = [
            python_executable,
            script,
            "--dataset",
            dataset,
            "--reduction_rate",
            ratio_text,
            "--gpu_id",
            "0",
            "--seed",
            str(seed),
            "--save",
            "1",
        ]
        return command, repo, {"script": script}

    if baseline == "ClustGDD":
        repo = baseline_root / "ClustGDD"
        script = "train_clustgdd_induct.py" if dataset == "reddit" else "train_clustgdd_transduct.py"
        command = [
            python_executable,
            script,
            "--dataset",
            dataset,
            "--reduction_rate",
            ratio_text,
            "--gpu_id",
            "0",
            "--seed",
            str(seed),
            "--save",
            "1",
        ]
        command.extend(_clustgdd_hyperparams(dataset))
        return command, repo, {"script": script}

    raise AssertionError(f"Unhandled baseline: {baseline}")


def _gecc_hyperparams(dataset: str) -> list[str]:
    common = ["--fuzziness", "1", "--rep_fuzz", "1", "--depth", "2"]
    if dataset == "ogbn-arxiv":
        return [
            "--agg_alpha",
            "0.5",
            "--agg_beta",
            "0.0",
            "--agg_gamma",
            "0.6",
            "--ewd",
            "0.001",
            *common,
        ]
    if dataset == "reddit":
        return [
            "--agg_alpha",
            "0.2",
            "--agg_beta",
            "0.9",
            "--agg_gamma",
            "0.7",
            "--ewd",
            "0.0005",
            *common,
        ]
    if dataset == "ogbn-products":
        return [
            "--agg_alpha",
            "0.1",
            "--agg_beta",
            "0.0",
            "--agg_gamma",
            "0.9",
            "--ewd",
            "0.001",
            *common,
        ]
    return common


def _clustgdd_hyperparams(dataset: str) -> list[str]:
    if dataset == "reddit":
        return ["--prop_num", "10", "--postprop_num", "7"]
    if dataset == "ogbn-arxiv":
        return ["--prop_num", "20", "--postprop_num", "10"]
    return []


def tgcc_config_for_plan(plan: RunPlan) -> dict[str, dict[str, Any]]:
    section = plan.metadata["config_section"]
    run_dir_abs = plan.run_dir.expanduser().resolve()
    config = {
        "dataset": plan.dataset,
        "reduction_rate": plan.baseline_ratio,
        "seed": plan.seed,
        "method": "TGCC",
        "epochs": 600,
        "hidden": 256,
        "nlayers": 2,
        "weight_decay": 0.0,
        "dropout": 0.0,
        "normalize_features": True,
        "keep_ratio": 1.0,
        "save": 1,
        "sgc": 1,
        "inner": 0,
        "outer": 20,
        "marks": f"shadow-hgc-full-{ratio_label(plan.full_ratio)}",
        "log_dir": path_arg(run_dir_abs / "logs"),
        "save_dir": path_arg(run_dir_abs / "saved_ours"),
    }
    return {section: config}


def materialize_run_inputs(plan: RunPlan) -> None:
    plan.run_dir.mkdir(parents=True, exist_ok=True)
    if plan.status == "unsupported":
        return

    if plan.baseline == "DeepCGC":
        ensure_deepcgc_ratio_transfer_fallback(plan.cwd / "scr" / "utils.py")
        ensure_deepcgc_device_mask_compat(plan.cwd / "scr" / "utils.py")
        write_deepcgc_config(plan)

    if plan.baseline == "GECC":
        ensure_gecc_reddit_graphsaint_loader(plan.cwd / "dataset" / "loader.py")

    if plan.baseline in {"TGCC", "WbGC"}:
        ensure_data_link(plan.cwd / "data", plan.data_root)
    elif plan.baseline == "ClustGDD":
        ensure_data_link(plan.cwd / "data", plan.data_root)
        ensure_data_link(plan.cwd / f"xxx{plan.dataset}", plan.data_root / plan.dataset)

    if plan.baseline == "TGCC":
        config_path = Path(plan.metadata["config_path"])
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(tgcc_config_for_plan(plan), indent=2), encoding="utf-8")
    elif plan.baseline == "WbGC":
        ensure_numpy_legacy_alias_compat(plan.cwd / "utils_graphsaint.py")
        ensure_wbgc_graphsaint_sampler_sizes(plan.cwd / "utils_graphsaint.py")
    elif plan.baseline == "ClustGDD":
        ensure_numpy_legacy_alias_compat(
            plan.cwd / "deep_robust_utils.py",
            plan.cwd / "deep_robust_data.py",
        )
        ensure_clustgdd_induct_import_compat(plan.cwd / "clustgdd_agent_induct.py")


def ensure_gecc_reddit_graphsaint_loader(loader_path: Path) -> None:
    marker = "# Shadow-HGC launcher patch: use local GraphSAINT Reddit data"
    text = loader_path.read_text(encoding="utf-8")
    if marker in text:
        return
    old = """        elif name in ['reddit']:
            dataset = Reddit2(root=path + '/reddit')
"""
    new = """        elif name in ['reddit']:
            {marker}
            dataset = DataGraphSAINT(root=path, dataset=name)
            dataset.num_classes = 41
""".format(marker=marker)
    if old not in text:
        raise RuntimeError(f"Could not patch GECC Reddit loader in {loader_path}")
    loader_path.write_text(text.replace(old, new), encoding="utf-8")


def ensure_deepcgc_ratio_transfer_fallback(utils_path: Path) -> None:
    marker = "# Shadow-HGC launcher patch: arbitrary arxiv train-ratio fallback"
    text = utils_path.read_text(encoding="utf-8")
    if marker in text:
        return
    old = """        if args.ratio == 0.005:
            return 0.01

    else:
        return args.ratio
"""
    new = """        if args.ratio == 0.005:
            return 0.01
        {marker}
        return args.ratio

    else:
        return args.ratio
""".format(marker=marker)
    if old not in text:
        raise RuntimeError(f"Could not patch DeepCGC ratio_transfer fallback in {utils_path}")
    utils_path.write_text(text.replace(old, new), encoding="utf-8")


def ensure_deepcgc_device_mask_compat(utils_path: Path) -> None:
    text = utils_path.read_text(encoding="utf-8")
    patched = text.replace("idx[cls_mask][center_mask]", "idx[cls_mask.cpu()][center_mask]")
    patched = patched.replace("idx[cls_mask[cls]][center_mask]", "idx[cls_mask[cls].cpu()][center_mask]")
    if patched != text:
        utils_path.write_text(patched, encoding="utf-8")


def ensure_numpy_legacy_alias_compat(*paths: Path) -> None:
    for path in paths:
        text = path.read_text(encoding="utf-8")
        patched = re.sub(r"\bnp\.int\b", "int", text)
        patched = re.sub(r"\bnp\.bool\b", "bool", patched)
        if patched != text:
            path.write_text(patched, encoding="utf-8")


def ensure_wbgc_graphsaint_sampler_sizes(utils_path: Path) -> None:
    text = utils_path.read_text(encoding="utf-8")
    if "if args.nlayers == 3:" in text:
        return
    patched = re.sub(
        r"(            else:\n                sizes = \[10, 5\]\n)\s*(        if self\.class_dict2 is None:\n)",
        r"\1        if args.nlayers == 3:\n            sizes = [15, 10, 5]\n\n\2",
        text,
        count=1,
    )
    if patched == text:
        raise RuntimeError(f"Could not patch WbGC GraphSAINT sampler sizes in {utils_path}")
    utils_path.write_text(patched, encoding="utf-8")


def ensure_clustgdd_induct_import_compat(agent_path: Path) -> None:
    text = agent_path.read_text(encoding="utf-8")
    old = "from KDD2025_ClustGDD.utils_clustgdd import graph_analysis, ER_estimator, attaw_ER_estimator"
    new = "from utils_clustgdd import graph_analysis, ER_estimator, attaw_ER_estimator"
    if old in text:
        agent_path.write_text(text.replace(old, new), encoding="utf-8")


def write_deepcgc_config(plan: RunPlan) -> None:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - DeepCGC itself requires yaml.
        raise RuntimeError("DeepCGC config generation requires PyYAML") from exc

    dataset_arg = plan.metadata["dataset_arg"]
    source_path = plan.cwd / "config" / f"{dataset_arg}.yaml"
    config_folder = Path(plan.metadata["config_folder"])
    config_folder.mkdir(parents=True, exist_ok=True)
    with source_path.open("r", encoding="utf-8") as handle:
        source = yaml.safe_load(handle)

    dataset_config = source[dataset_arg]
    nearest_ratio_key = min(
        (key for key in dataset_config if key.startswith("ratio:")),
        key=lambda key: abs(float(key.split(":", 1)[1]) - plan.baseline_ratio),
    )
    generate_adj_key = "generate_adj:0"
    if generate_adj_key not in dataset_config[nearest_ratio_key]:
        generate_adj_key = next(iter(dataset_config[nearest_ratio_key]))
    hyperparams = dataset_config[nearest_ratio_key][generate_adj_key]
    generated = {
        dataset_arg: {
            f"ratio:{_fmt_ratio(plan.baseline_ratio)}": {
                "generate_adj:0": hyperparams,
            }
        }
    }
    with (config_folder / f"{dataset_arg}.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(generated, handle, sort_keys=False)


def ensure_data_link(link_path: Path, target_path: Path) -> None:
    link_path = Path(link_path)
    target_path = Path(target_path)
    if link_path.exists() or link_path.is_symlink():
        if link_path.is_symlink():
            resolved = link_path.resolve()
            if resolved != target_path.resolve():
                link_path.unlink()
            else:
                return
        else:
            return
    if link_path.exists():
        return
    link_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        link_path.symlink_to(target_path, target_is_directory=True)
    except OSError:
        if os.name != "nt":
            raise
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link_path), str(target_path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def directory_size_bytes(path: Path) -> int:
    path = Path(path)
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file() and not item.is_symlink():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def parse_metrics_from_text(text: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    accuracy = _first_float(
        text,
        [
            rf"Average\s+accuracy\s*[:=]\s*({_FLOAT_RE})",
            rf"['\"]?acc[_\s-]?test['\"]?\s*[:=]\s*({_FLOAT_RE})",
            rf"['\"]?test[_\s-]?acc(?:uracy)?['\"]?\s*[:=]\s*({_FLOAT_RE})",
            rf"['\"]?accuracy['\"]?\s*[:=]\s*({_FLOAT_RE})",
            rf"\bacc\b\s*[:=]\s*({_FLOAT_RE})",
        ],
    )
    if accuracy is not None:
        metrics["accuracy"] = _as_fraction(accuracy)

    micro = _first_float(
        text,
        [
            rf"\bmicro[_\s-]?f1\b\s*[:=]\s*({_FLOAT_RE})",
            rf"\bf1[_\s-]?micro\b\s*[:=]\s*({_FLOAT_RE})",
            rf"\bF1[_\s-]?micro\b\s*[:=]\s*({_FLOAT_RE})",
        ],
    )
    if micro is not None:
        metrics["micro_f1"] = _as_fraction(micro)

    macro = _first_float(
        text,
        [
            rf"\bmacro[_\s-]?f1\b\s*[:=]\s*({_FLOAT_RE})",
            rf"\bf1[_\s-]?macro\b\s*[:=]\s*({_FLOAT_RE})",
            rf"\bF1[_\s-]?macro\b\s*[:=]\s*({_FLOAT_RE})",
        ],
    )
    if macro is not None:
        metrics["macro_f1"] = _as_fraction(macro)

    param_count = _first_int(
        text,
        [
            r"(?:Total\s+)?(?:parameters|params|param_count|num_params)\s*[:=]\s*([0-9][0-9,]*)",
            r"model\s+(?:parameters|params)\s*[:=]\s*([0-9][0-9,]*)",
        ],
    )
    if param_count is not None:
        metrics["param_count"] = param_count

    return metrics


def _first_float(text: str, patterns: Iterable[str]) -> float | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    return None


def _first_int(text: str, patterns: Iterable[str]) -> int | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            try:
                return int(match.group(1).replace(",", ""))
            except ValueError:
                continue
    return None


def _as_fraction(value: float) -> float:
    if math.isfinite(value) and 1.0 < value <= 100.0:
        return value / 100.0
    return value


def detect_failure_status(returncode: int | None, text: str) -> tuple[str, str]:
    lower = text.lower().replace(" ", "")
    lower_spaced = text.lower()
    for marker, reason in _OOM_MARKERS:
        compact_marker = marker.replace(" ", "")
        if compact_marker in lower or marker in lower_spaced:
            return "oom", reason
    if returncode in {137, -9}:
        return "oom", "process killed, likely out of memory"
    if returncode in {124, -15}:
        return "timeout", "process timed out or was terminated"
    if returncode == 0:
        return "completed", ""
    return "failed", f"return code {returncode}"


def plan_to_record(plan: RunPlan) -> dict[str, Any]:
    return {
        "baseline": plan.baseline,
        "dataset": plan.dataset,
        "schedule_dataset": plan.schedule_dataset,
        "seed": plan.seed,
        "gpu": plan.gpu,
        "requested_full_node_ratio": plan.full_ratio,
        "planned_condensed_nodes": plan.planned_condensed_nodes,
        "baseline_ratio": plan.baseline_ratio,
        "baseline_ratio_denominator": plan.baseline_ratio_denominator,
        "status": plan.status,
        "failure_reason": plan.failure_reason,
        "cwd": path_arg(plan.cwd),
        "command": plan.command,
        "env": plan.env,
        "data_root": path_arg(plan.data_root),
        "baseline_root": path_arg(plan.baseline_root),
        "run_dir": path_arg(plan.run_dir),
        "stdout_path": path_arg(plan.stdout_path),
        "stderr_path": path_arg(plan.stderr_path),
        "metrics_path": path_arg(plan.metrics_path),
        "summary_path": path_arg(plan.summary_path),
        "micro_f1": None,
        "macro_f1": None,
        "accuracy": None,
        "param_count": None,
        "storage_bytes": 0,
        "storage_measured_path": path_arg(plan.run_dir),
        "metadata": plan.metadata,
    }


def execute_plan(plan: RunPlan, timeout_sec: int | None = None) -> dict[str, Any]:
    materialize_run_inputs(plan)
    record = plan_to_record(plan)
    if plan.status == "unsupported":
        record["storage_bytes"] = directory_size_bytes(plan.run_dir)
        write_run_record(plan, record)
        return record

    preflight_reason = preflight_failure_reason(plan)
    if preflight_reason:
        record["status"] = "failed"
        record["failure_reason"] = preflight_reason
        record["returncode"] = None
        record["storage_bytes"] = directory_size_bytes(plan.run_dir)
        write_run_record(plan, record)
        return record

    env = os.environ.copy()
    env.update(plan.env)
    record["status"] = "running"
    record["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    start = time.perf_counter()

    try:
        with plan.stdout_path.open("w", encoding="utf-8", errors="replace") as stdout_file, plan.stderr_path.open(
            "w", encoding="utf-8", errors="replace"
        ) as stderr_file:
            proc = subprocess.run(
                plan.command,
                cwd=plan.cwd,
                env=env,
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
                timeout=timeout_sec,
            )
        stdout_text = _read_text(plan.stdout_path)
        stderr_text = _read_text(plan.stderr_path)
        combined_text = f"{stdout_text}\n{stderr_text}"
        status, reason = detect_failure_status(proc.returncode, combined_text)
        record["returncode"] = proc.returncode
    except subprocess.TimeoutExpired as exc:
        combined_text = f"{_read_text(plan.stdout_path)}\n{_read_text(plan.stderr_path)}\n{exc}"
        status, reason = "timeout", f"timeout after {timeout_sec} seconds"
        record["returncode"] = None
    except Exception as exc:  # keep unexpected launcher failures machine-readable
        combined_text = f"{_read_text(plan.stdout_path)}\n{_read_text(plan.stderr_path)}\n{type(exc).__name__}: {exc}"
        status, reason = detect_failure_status(None, combined_text)
        if status == "completed":
            status = "failed"
            reason = f"{type(exc).__name__}: {exc}"
        record["returncode"] = None

    metrics = parse_metrics_from_text(combined_text)
    if status == "completed" and metrics.get("accuracy") is None and metrics.get("micro_f1") is None:
        status = "failed"
        reason = _missing_metric_failure_reason(combined_text)
    if "accuracy" in metrics and "micro_f1" not in metrics:
        metrics["micro_f1"] = metrics["accuracy"]
        metrics["micro_f1_inferred_from_accuracy"] = True

    record.update(metrics)
    record["status"] = status
    record["failure_reason"] = reason
    record["elapsed_sec"] = time.perf_counter() - start
    record["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    record["storage_bytes"] = directory_size_bytes(plan.run_dir)
    write_run_record(plan, record)
    record["storage_bytes"] = directory_size_bytes(plan.run_dir)
    write_run_record(plan, record)
    return record


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def _missing_metric_failure_reason(text: str) -> str:
    if re.search(r"\b\d+(?:st|nd|rd|th)\s+\(\d+\)\s+class:\s+\d+", text):
        return "completed without metrics: class budget mismatch"
    return "completed without parseable accuracy/micro-F1"


def write_run_record(plan: RunPlan, record: dict[str, Any]) -> None:
    plan.run_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "accuracy": record.get("accuracy"),
        "micro_f1": record.get("micro_f1"),
        "macro_f1": record.get("macro_f1"),
        "param_count": record.get("param_count"),
        "storage_bytes": record.get("storage_bytes"),
        "status": record.get("status"),
        "failure_reason": record.get("failure_reason"),
    }
    plan.metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plan.summary_path.write_text(json.dumps(record, indent=2), encoding="utf-8")


def preflight_failure_reason(plan: RunPlan) -> str:
    if plan.baseline == "TGCC":
        missing = tgcc_missing_augmentation_files(plan)
        if missing:
            return "missing TGCC precomputed augmentation file(s): " + ", ".join(path_arg(path) for path in missing)
    return ""


def tgcc_missing_augmentation_files(plan: RunPlan) -> list[Path]:
    if plan.baseline != "TGCC":
        return []
    if plan.dataset == "reddit":
        required = [plan.cwd / "data" / plan.dataset / "0.01_1_1tr.npz"]
    elif plan.dataset == "ogbn-arxiv":
        required = [plan.cwd / "data" / plan.dataset / "0.01_1_0.npz"]
    else:
        required = []
    return [path for path in required if not path.exists()]


def write_records_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def write_records_csv(path: Path, records: Iterable[dict[str, Any]]) -> None:
    rows = list(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    preferred = [
        "baseline",
        "dataset",
        "schedule_dataset",
        "seed",
        "gpu",
        "requested_full_node_ratio",
        "planned_condensed_nodes",
        "baseline_ratio",
        "baseline_ratio_denominator",
        "status",
        "accuracy",
        "micro_f1",
        "macro_f1",
        "param_count",
        "storage_bytes",
        "failure_reason",
        "elapsed_sec",
        "returncode",
        "run_dir",
        "command",
    ]
    remaining = sorted({key for row in rows for key in row} - set(preferred))
    fieldnames = [name for name in preferred if any(name in row for row in rows)] + remaining
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=True)
    return value
