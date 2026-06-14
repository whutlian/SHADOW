from __future__ import annotations

from dataclasses import dataclass

from shadow_hgc.sft.unified_schedule import compute_unified_schedule


@dataclass(frozen=True)
class STTGatedMixerConfig:
    input_dim: int
    num_classes: int
    hidden_dim: int
    epochs: int
    internal_style: str
    dropout: float = 0.10


def make_stt_gated_mixer_config(
    *,
    input_dim: int,
    num_classes: int,
    condensed_nodes: int,
    teacher_valid_acc: float | None = None,
    majority_valid_acc: float | None = None,
    num_nodes: int = 1,
) -> STTGatedMixerConfig:
    schedule = compute_unified_schedule(
        condensed_nodes=int(condensed_nodes),
        num_classes=int(num_classes),
        teacher_valid_acc=teacher_valid_acc,
        majority_valid_acc=majority_valid_acc,
        num_nodes=int(num_nodes),
        num_teacher_nodes=int(num_nodes),
    )
    return STTGatedMixerConfig(
        input_dim=int(input_dim),
        num_classes=int(num_classes),
        hidden_dim=schedule.hidden_dim,
        epochs=schedule.epochs,
        internal_style=schedule.student_internal_style,
    )


def estimate_stt_gated_mixer_params(*, input_dim: int, num_classes: int, hidden_dim: int) -> int:
    # Gated table mixer proxy used for reporting: input projection, two gates, classifier, and biases.
    return int(input_dim) * int(hidden_dim) * 3 + int(hidden_dim) * int(num_classes) + int(hidden_dim) * 3 + int(num_classes)
