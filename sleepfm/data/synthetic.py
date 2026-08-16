"""Synthetic multi-modal sleep epochs for demo training without PSG access."""



from __future__ import annotations



import json

from pathlib import Path

from typing import Dict, List, Optional, Tuple



import numpy as np



SLEEP_STAGES = ["Wake", "Stage 1", "Stage 2", "Stage 3", "REM"]

STAGE_TO_ID = {s: i for i, s in enumerate(SLEEP_STAGES)}





def _stage_waveform(stage_id: int, length: int, rng: np.random.Generator) -> np.ndarray:

    """Simple stage-dependent frequency content for BAS-like signal."""

    t = np.linspace(0, 1, length, endpoint=False)

    freqs = {0: 12.0, 1: 6.0, 2: 4.0, 3: 2.0, 4: 8.0}

    f = freqs.get(stage_id, 4.0)

    return np.sin(2 * np.pi * f * t) + 0.1 * rng.standard_normal(length)





def generate_epoch_signals(

    channels: Dict[str, int],

    clip_length: int,

    stage_id: int,

    apnea: bool,

    rng: np.random.Generator,

    participant_bias: float = 0.0,

) -> Dict[str, np.ndarray]:

    """Generate correlated multi-modal 30s (or shorter) clips."""

    bas_base = _stage_waveform(stage_id, clip_length, rng) + participant_bias

    signals = {}

    for ch in range(channels["bas"]):

        signals.setdefault("bas", []).append(

            bas_base + 0.05 * ch * rng.standard_normal(clip_length)

        )

    signals["bas"] = np.stack(signals["bas"], axis=0).astype(np.float32)



    hr = 1.2 + 0.05 * stage_id + 0.02 * participant_bias

    t = np.arange(clip_length) / clip_length

    ecg = np.sin(2 * np.pi * hr * 30 * t)

    if apnea:

        ecg = ecg * (1.0 - 0.5 * (t > 0.5))

    signals["ecg"] = np.stack(

        [ecg + 0.02 * rng.standard_normal(clip_length) for _ in range(channels["ecg"])],

        axis=0,

    ).astype(np.float32)



    resp = np.abs(np.sin(2 * np.pi * 0.25 * t))

    if apnea:

        resp = resp * (t < 0.5).astype(float)

    signals["respiratory"] = np.stack(

        [

            resp * (1 + 0.1 * i) + 0.03 * rng.standard_normal(clip_length)

            for i in range(channels["respiratory"])

        ],

        axis=0,

    ).astype(np.float32)

    return signals





def generate_synthetic_split(

    num_epochs: int,

    channels: Dict[str, int],

    clip_length: int,

    seed: int = 42,

    apnea_rate: float = 0.15,

    participant_id: Optional[str] = None,

    participant_bias: float = 0.0,

) -> List[dict]:

    rng = np.random.default_rng(seed)

    records = []

    for i in range(num_epochs):

        stage_id = int(rng.integers(0, len(SLEEP_STAGES)))

        apnea = bool(rng.random() < apnea_rate)

        signals = generate_epoch_signals(

            channels,

            clip_length,

            stage_id,

            apnea,

            rng,

            participant_bias=participant_bias,

        )

        rec = {

            "epoch_id": i,

            "stage": SLEEP_STAGES[stage_id],

            "stage_id": stage_id,

            "apnea": int(apnea),

            "signals": signals,

        }

        if participant_id is not None:

            rec["participant_id"] = participant_id

        records.append(rec)

    return records





def assign_participants_to_splits(

    num_participants: int,

    split_counts: Dict[str, int],

    seed: int = 42,

) -> Dict[str, List[str]]:

    """

    Assign unique participant IDs to splits (no participant spans two splits).



    split_counts values are interpreted as *target epoch counts*; participants

    are allocated proportionally, with at least one participant per non-empty split.

    """

    rng = np.random.default_rng(seed)

    pids = [f"P{idx:04d}" for idx in range(num_participants)]

    rng.shuffle(pids)



    total_epochs = sum(split_counts.values())

    if total_epochs <= 0:

        raise ValueError("split_counts must sum to a positive epoch count")



    active = [s for s, c in split_counts.items() if c > 0]

    if num_participants < len(active):

        raise ValueError(

            f"Need at least {len(active)} participants for splits {active}, got {num_participants}"

        )



    # Proportional participant allocation

    raw = {s: split_counts[s] / total_epochs * num_participants for s in active}

    alloc = {s: max(1, int(round(raw[s]))) for s in active}

    while sum(alloc.values()) > num_participants:

        richest = max(active, key=lambda s: alloc[s])

        if alloc[richest] > 1:

            alloc[richest] -= 1

        else:

            break

    while sum(alloc.values()) < num_participants:

        neediest = max(active, key=lambda s: split_counts[s] - alloc[s])

        alloc[neediest] += 1



    out: Dict[str, List[str]] = {s: [] for s in split_counts}

    idx = 0

    for split in split_counts:

        n_p = alloc.get(split, 0) if split_counts[split] > 0 else 0

        out[split] = pids[idx : idx + n_p]

        idx += n_p

    # Remaining participants go to largest split

    if idx < num_participants:

        largest = max(active, key=lambda s: split_counts[s])

        out[largest].extend(pids[idx:])

    return out





def write_synthetic_dataset(

    output_dir: str | Path,

    channels: Dict[str, int],

    clip_length: int,

    splits: Dict[str, int],

    seed: int = 42,

    apnea_rate: float = 0.15,

    num_participants: Optional[int] = None,

    epochs_per_participant: Optional[int] = None,

) -> Path:

    """

    Write .npy epochs + index.json per split.



    When num_participants is set, epochs are grouped by participant and each

    participant belongs to exactly one split (paper-style cohort isolation).

    """

    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)



    if num_participants is None:

        return _write_epoch_level_dataset(

            output_dir, channels, clip_length, splits, seed, apnea_rate

        )



    if epochs_per_participant is None:

        total = sum(splits.values())

        epochs_per_participant = max(1, (total + num_participants - 1) // num_participants)



    participant_map = assign_participants_to_splits(num_participants, splits, seed=seed)

    index: Dict[str, list] = {}

    global_epoch = 0



    for split, target_count in splits.items():

        if target_count <= 0:

            index[split] = []

            continue

        split_entries = []

        pids = participant_map.get(split, [])

        if not pids:

            continue

        epochs_written = 0

        for pi, pid in enumerate(pids):

            bias = (hash(pid) % 1000) / 500.0 - 1.0

            n_ep = epochs_per_participant

            if epochs_written + n_ep > target_count:

                n_ep = target_count - epochs_written

            if n_ep <= 0:

                continue

            records = generate_synthetic_split(

                n_ep,

                channels,

                clip_length,

                seed=seed + global_epoch + pi,

                apnea_rate=apnea_rate,

                participant_id=pid,

                participant_bias=bias,

            )

            for rec in records:

                fname = f"{split}_{global_epoch:06d}.npy"

                path = output_dir / fname

                stacked = np.concatenate(

                    [

                        rec["signals"]["bas"],

                        rec["signals"]["ecg"],

                        rec["signals"]["respiratory"],

                    ],

                    axis=0,

                )

                np.save(path, stacked)

                split_entries.append(

                    {

                        "path": path.name,

                        "stage": rec["stage"],

                        "stage_id": rec["stage_id"],

                        "apnea": rec["apnea"],

                        "participant_id": pid,

                        "epoch_index": rec["epoch_id"],

                        "night_id": pid,

                    }

                )

                global_epoch += 1

                epochs_written += 1

                if epochs_written >= target_count:

                    break

            if epochs_written >= target_count:

                break

        index[split] = split_entries



    meta = _build_meta(channels, clip_length, num_participants=num_participants)

    with open(output_dir / "index.json", "w", encoding="utf-8") as f:

        json.dump({"meta": meta, "splits": index}, f, indent=2)

    return output_dir





def _write_epoch_level_dataset(

    output_dir: Path,

    channels: Dict[str, int],

    clip_length: int,

    splits: Dict[str, int],

    seed: int,

    apnea_rate: float,

) -> Path:

    """Legacy epoch-level generator (unique paths per split, no participant_id)."""

    index = {}

    offset = 0

    for split, count in splits.items():

        records = generate_synthetic_split(

            count, channels, clip_length, seed=seed + offset, apnea_rate=apnea_rate

        )

        offset += count

        split_entries = []

        for rec in records:

            fname = f"{split}_{rec['epoch_id']:06d}.npy"

            path = output_dir / fname

            stacked = np.concatenate(

                [rec["signals"]["bas"], rec["signals"]["ecg"], rec["signals"]["respiratory"]],

                axis=0,

            )

            np.save(path, stacked)

            split_entries.append(

                {

                    "path": str(path.name),

                    "stage": rec["stage"],

                    "stage_id": rec["stage_id"],

                    "apnea": rec["apnea"],

                }

            )

        index[split] = split_entries



    meta = _build_meta(channels, clip_length, num_participants=None)

    with open(output_dir / "index.json", "w", encoding="utf-8") as f:

        json.dump({"meta": meta, "splits": index}, f, indent=2)

    return output_dir





def _build_meta(

    channels: Dict[str, int],

    clip_length: int,

    num_participants: Optional[int],

) -> dict:

    meta = {

        "channels": channels,

        "clip_length": clip_length,

        "channel_order": ["bas", "ecg", "respiratory"],

        "channel_slices": {

            "bas": [0, channels["bas"]],

            "ecg": [channels["bas"], channels["bas"] + channels["ecg"]],

            "respiratory": [

                channels["bas"] + channels["ecg"],

                channels["bas"] + channels["ecg"] + channels["respiratory"],

            ],

        },

        "synthetic": True,

    }

    if num_participants is not None:

        meta["num_participants"] = num_participants

        meta["participant_level_splits"] = True

    return meta


