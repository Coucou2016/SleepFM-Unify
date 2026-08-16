"""Export EDF / WFDB-MAT / NPZ PSG recordings to SleepFM index.json + .npy epochs.

Does not change the on-disk schema in docs/DATA_SCHEMA.md. Unmatched channels
are zero-padded so BAS/ECG/respiratory widths stay at the configured counts
(default 10 / 2 / 7).
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from xml.etree import ElementTree as ET

import numpy as np

from sleepfm.data.channel_map import (
    MODALITY_ORDER,
    ChannelTable,
    document_missing,
    load_channel_table,
    map_recording_channels,
)
from sleepfm.data.splits import assign_participant_splits
from sleepfm.data.synthetic import SLEEP_STAGES

try:
    from scipy.signal import resample_poly
except ImportError:  # pragma: no cover
    resample_poly = None

STAGE_ALIASES = {
    "w": 0,
    "wake": 0,
    "wakefulness": 0,
    "sleep stage w": 0,
    "stage 0": 0,
    "n1": 1,
    "stage 1": 1,
    "nrem1": 1,
    "nrem 1": 1,
    "sleep stage 1": 1,
    "stage1": 1,
    "n2": 2,
    "stage 2": 2,
    "nrem2": 2,
    "nrem 2": 2,
    "sleep stage 2": 2,
    "stage2": 2,
    "n3": 3,
    "stage 3": 3,
    "stage 4": 3,
    "nrem3": 3,
    "nrem 3": 3,
    "sleep stage 3": 3,
    "sleep stage 4": 3,
    "slow wave": 3,
    "sws": 3,
    "r": 4,
    "rem": 4,
    "sleep stage r": 4,
    "sleep stage rem": 4,
    "mt": 0,
    "movement": 0,
    "unscored": 0,
    "unknown": 0,
}

APNEA_KEYWORDS = (
    "apnea",
    "apnoea",
    "hypopnea",
    "hypopnoea",
    "obstructive apnea",
    "central apnea",
    "mixed apnea",
    "respiratory event",
)


@dataclass
class Recording:
    recording_id: str
    participant_id: str
    signals: Dict[str, np.ndarray]
    channel_fs: Dict[str, float]
    annotations: List[dict] = field(default_factory=list)
    source_path: Optional[str] = None


def _ascii_field(value: str, width: int) -> bytes:
    raw = str(value).encode("ascii", errors="replace")[:width]
    return raw + b" " * (width - len(raw))


def write_edf_basic(
    path: str | Path,
    signals: Dict[str, np.ndarray],
    fs: int,
    patient: str = "",
    recording: str = "SleepFM fixture",
) -> Path:
    """Write a minimal 16-bit EDF (shared sample rate). Used for tests/fixtures."""
    path = Path(path)
    names = list(signals.keys())
    if not names:
        raise ValueError("No signals to write")
    arrays = []
    n_samples = None
    for name in names:
        x = np.asarray(signals[name], dtype=np.float64).reshape(-1)
        arrays.append(x)
        n_samples = len(x) if n_samples is None else min(n_samples, len(x))
    assert n_samples is not None
    arrays = [a[:n_samples] for a in arrays]
    record_duration = 1.0
    samples_per_record = int(fs)
    n_records = max(1, n_samples // samples_per_record)
    used = n_records * samples_per_record
    arrays = [a[:used] for a in arrays]

    n_sig = len(names)
    header_bytes = 256 + 256 * n_sig
    header = bytearray()
    header += _ascii_field("0", 8)
    header += _ascii_field(patient, 80)
    header += _ascii_field(recording, 80)
    header += _ascii_field("01.01.00", 8)
    header += _ascii_field("00.00.00", 8)
    header += _ascii_field(str(header_bytes), 8)
    header += _ascii_field("", 44)
    header += _ascii_field(str(n_records), 8)
    header += _ascii_field(str(int(record_duration)), 8)
    header += _ascii_field(str(n_sig), 4)
    for name in names:
        header += _ascii_field(name, 16)
    for _ in names:
        header += _ascii_field("", 80)
    for _ in names:
        header += _ascii_field("uV", 8)
    phys_params = []
    for arr in arrays:
        pmin = float(np.min(arr)) if arr.size else -1.0
        pmax = float(np.max(arr)) if arr.size else 1.0
        if pmin == pmax:
            pmax = pmin + 1.0
        phys_params.append((pmin, pmax))
    for pmin, _ in phys_params:
        header += _ascii_field(f"{pmin:.6g}", 8)
    for _, pmax in phys_params:
        header += _ascii_field(f"{pmax:.6g}", 8)
    for _ in names:
        header += _ascii_field("-32768", 8)
    for _ in names:
        header += _ascii_field("32767", 8)
    for _ in names:
        header += _ascii_field("", 80)
    for _ in names:
        header += _ascii_field(str(samples_per_record), 8)
    for _ in names:
        header += _ascii_field("", 32)
    if len(header) != header_bytes:
        raise RuntimeError(f"EDF header size {len(header)} != {header_bytes}")

    dmin, dmax = -32768.0, 32767.0
    body = bytearray()
    for rec_i in range(n_records):
        sl = slice(rec_i * samples_per_record, (rec_i + 1) * samples_per_record)
        for arr, (pmin, pmax) in zip(arrays, phys_params):
            scaled = (arr[sl] - pmin) / (pmax - pmin) * (dmax - dmin) + dmin
            scaled = np.clip(np.rint(scaled), dmin, dmax).astype("<i2")
            body += scaled.tobytes()

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(header) + bytes(body))
    return path


def _read_ascii(buf: bytes, start: int, width: int) -> str:
    return buf[start : start + width].decode("ascii", errors="replace").strip()


def read_edf_basic(path: str | Path) -> Recording:
    """Read a standard (non-EDF+) 16-bit EDF into a Recording."""
    path = Path(path)
    data = path.read_bytes()
    n_sig = int(_read_ascii(data, 252, 4))
    header_bytes = int(_read_ascii(data, 184, 8))
    n_records = int(_read_ascii(data, 236, 8))
    rec_dur = float(_read_ascii(data, 244, 8) or "1")
    patient = _read_ascii(data, 8, 80)
    off = 256
    labels = [_read_ascii(data, off + 16 * i, 16) for i in range(n_sig)]
    off = 256 + 16 * n_sig + 80 * n_sig + 8 * n_sig
    phys_min = [float(_read_ascii(data, off + 8 * i, 8) or "0") for i in range(n_sig)]
    off += 8 * n_sig
    phys_max = [float(_read_ascii(data, off + 8 * i, 8) or "1") for i in range(n_sig)]
    off += 8 * n_sig
    dig_min = [float(_read_ascii(data, off + 8 * i, 8) or "-32768") for i in range(n_sig)]
    off += 8 * n_sig
    dig_max = [float(_read_ascii(data, off + 8 * i, 8) or "32767") for i in range(n_sig)]
    off += 8 * n_sig + 80 * n_sig
    n_samp = [int(_read_ascii(data, off + 8 * i, 8) or "0") for i in range(n_sig)]
    body = memoryview(data)[header_bytes:]
    samples: List[List[int]] = [[] for _ in range(n_sig)]
    cursor = 0
    for _ in range(max(n_records, 0)):
        for i in range(n_sig):
            n = n_samp[i]
            nbytes = n * 2
            chunk = body[cursor : cursor + nbytes]
            samples[i].extend(struct.unpack("<" + "h" * n, chunk))
            cursor += nbytes

    signals: Dict[str, np.ndarray] = {}
    channel_fs: Dict[str, float] = {}
    for i, name in enumerate(labels):
        digital = np.asarray(samples[i], dtype=np.float64)
        dspan = dig_max[i] - dig_min[i]
        pspan = phys_max[i] - phys_min[i]
        if dspan == 0:
            phys = np.zeros_like(digital)
        else:
            phys = (digital - dig_min[i]) / dspan * pspan + phys_min[i]
        key = name or f"ch{i}"
        signals[key] = phys.astype(np.float32)
        channel_fs[key] = (n_samp[i] / rec_dur) if rec_dur else float(n_samp[i])

    pid = patient.split()[0] if patient else path.stem
    return Recording(
        recording_id=path.stem,
        participant_id=pid or path.stem,
        signals=signals,
        channel_fs=channel_fs,
        source_path=str(path),
    )


def read_edf(path: str | Path) -> Recording:
    path = Path(path)
    try:
        return read_edf_basic(path)
    except Exception as basic_exc:
        last = basic_exc
    try:
        import mne  # type: ignore

        raw = mne.io.read_raw_edf(str(path), preload=True, verbose="ERROR")
        signals = {ch: raw.get_data(picks=[ch])[0].astype(np.float32) for ch in raw.ch_names}
        fs = {ch: float(raw.info["sfreq"]) for ch in raw.ch_names}
        return Recording(path.stem, path.stem, signals, fs, source_path=str(path))
    except Exception as exc:
        last = exc
    try:
        import pyedflib  # type: ignore

        reader = pyedflib.EdfReader(str(path))
        try:
            signals = {}
            fs = {}
            for i in range(reader.signals_in_file):
                name = reader.getLabel(i) or f"ch{i}"
                signals[name] = reader.readSignal(i).astype(np.float32)
                fs[name] = float(reader.getSampleFrequency(i))
            return Recording(path.stem, path.stem, signals, fs, source_path=str(path))
        finally:
            reader.close()
    except Exception as exc:
        last = exc
    raise RuntimeError(
        f"Could not read EDF {path}. Tried built-in reader, mne, and pyedflib. Last error: {last}"
    )


def read_npz_recording(path: str | Path) -> Recording:
    path = Path(path)
    payload = np.load(path, allow_pickle=True)
    signals_obj = payload["signals"].item() if payload["signals"].shape == () else payload["signals"]
    if isinstance(signals_obj, dict):
        signals = {k: np.asarray(v, dtype=np.float32).reshape(-1) for k, v in signals_obj.items()}
    else:
        raise ValueError(f"{path}: 'signals' must be a dict of channel arrays")
    if "channel_fs" in payload:
        raw_fs = payload["channel_fs"]
        channel_fs = raw_fs.item() if getattr(raw_fs, "shape", ()) == () else dict(raw_fs)
        channel_fs = {k: float(v) for k, v in channel_fs.items()}
    else:
        fs0 = float(payload["fs"]) if "fs" in payload else 256.0
        channel_fs = {k: fs0 for k in signals}
    pid = str(payload["participant_id"]) if "participant_id" in payload else path.stem
    rid = str(payload["recording_id"]) if "recording_id" in payload else path.stem
    annotations: List[dict] = []
    if "annotations" in payload:
        raw_ann = payload["annotations"]
        annotations = list(raw_ann.tolist() if hasattr(raw_ann, "tolist") else raw_ann)
    return Recording(rid, pid, signals, channel_fs, annotations, source_path=str(path))


def write_npz_recording(path: str | Path, rec: Recording) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        signals=rec.signals,
        channel_fs=rec.channel_fs,
        participant_id=rec.participant_id,
        recording_id=rec.recording_id,
        annotations=np.array(rec.annotations, dtype=object),
    )
    return path


def parse_wfdb_hea(path: str | Path) -> Tuple[str, int, float, int, List[str]]:
    """Parse a WFDB .hea header: record name, n_sig, fs, n_samples, channel names."""
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        raise ValueError(f"Empty header {path}")
    parts = lines[0].split()
    rec_name = parts[0]
    n_sig = int(parts[1])
    fs = float(parts[2])
    n_samples = int(parts[3]) if len(parts) > 3 else 0
    names: List[str] = []
    for line in lines[1:]:
        if not line.strip() or line.startswith("#"):
            continue
        bits = line.split()
        names.append(bits[-1] if bits else f"ch{len(names)}")
        if len(names) >= n_sig:
            break
    while len(names) < n_sig:
        names.append(f"ch{len(names)}")
    return rec_name, n_sig, fs, n_samples, names


def read_cinc_mat(mat_path: str | Path, hea_path: Optional[str | Path] = None) -> Recording:
    from scipy.io import loadmat

    mat_path = Path(mat_path)
    hea_path = Path(hea_path) if hea_path else mat_path.with_suffix(".hea")
    mat = loadmat(str(mat_path))
    val = None
    for key in ("val", "data", "signals"):
        if key in mat:
            val = np.asarray(mat[key])
            break
    if val is None:
        numeric = [k for k in mat if not k.startswith("_") and isinstance(mat[k], np.ndarray)]
        if not numeric:
            raise ValueError(f"No signal array in {mat_path}")
        val = np.asarray(mat[numeric[0]])
    if val.ndim != 2:
        raise ValueError(f"Expected 2D signal matrix in {mat_path}, got {val.shape}")
    names = [f"ch{i}" for i in range(val.shape[0])]
    fs = 200.0
    if hea_path.is_file():
        _, n_sig, fs, _, names = parse_wfdb_hea(hea_path)
        names = names[: val.shape[0]]
        while len(names) < val.shape[0]:
            names.append(f"ch{len(names)}")
    signals = {names[i]: val[i].astype(np.float32) for i in range(val.shape[0])}
    return Recording(
        recording_id=mat_path.stem,
        participant_id=mat_path.stem,
        signals=signals,
        channel_fs={n: fs for n in names},
        source_path=str(mat_path),
    )


def normalize_stage_label(label: str) -> int:
    key = " ".join(str(label).lower().replace("_", " ").replace("-", " ").split())
    key = key.replace("sleep stage", "sleep stage")
    if key in STAGE_ALIASES:
        return STAGE_ALIASES[key]
    for prefix in ("sleep stage ", "stage ", "scored event ", "aasm "):
        if key.startswith(prefix) and key[len(prefix) :] in STAGE_ALIASES:
            return STAGE_ALIASES[key[len(prefix) :]]
    if "rem" in key:
        return 4
    if "wake" in key or key == "w":
        return 0
    return 0


def is_apnea_label(label: str, event_type: str = "") -> bool:
    blob = f"{label} {event_type}".lower()
    return any(k in blob for k in APNEA_KEYWORDS)


def parse_nsrr_xml(path: str | Path) -> List[dict]:
    path = Path(path)
    tree = ET.parse(path)
    root = tree.getroot()
    events: List[dict] = []
    for node in root.iter():
        tag = node.tag.split("}")[-1]
        if tag != "ScoredEvent":
            continue
        fields = {c.tag.split("}")[-1]: (c.text or "").strip() for c in list(node)}
        try:
            start = float(fields.get("Start") or fields.get("start") or 0.0)
            duration = float(fields.get("Duration") or fields.get("duration") or 0.0)
        except ValueError:
            continue
        label = fields.get("EventConcept") or fields.get("Name") or ""
        event_type = fields.get("EventType") or ""
        events.append(
            {
                "start": start,
                "duration": duration,
                "label": label,
                "event_type": event_type,
            }
        )
    return events


def parse_annotation_csv(path: str | Path) -> List[dict]:
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        return []
    header = [h.strip().lower() for h in lines[0].split(",")]
    events: List[dict] = []
    for line in lines[1:]:
        if not line.strip():
            continue
        cols = [c.strip() for c in line.split(",")]
        row = {header[i]: cols[i] if i < len(cols) else "" for i in range(len(header))}
        try:
            start = float(row.get("start") or row.get("onset") or 0.0)
            duration = float(row.get("duration") or row.get("dur") or 30.0)
        except ValueError:
            continue
        events.append(
            {
                "start": start,
                "duration": duration,
                "label": row.get("label") or row.get("stage") or row.get("event") or "",
                "event_type": row.get("type") or row.get("event_type") or "",
            }
        )
    return events


def parse_annotation_json(path: str | Path) -> List[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("annotations") or payload.get("events") or []
    return list(payload)


def try_load_wfdb_arousal(record_base: Path) -> List[dict]:
    try:
        import wfdb  # type: ignore
    except ImportError:
        return []
    try:
        ann = wfdb.rdann(str(record_base), "arousal")
    except Exception:
        return []
    events = []
    fs = float(ann.fs or 200.0)
    samples = list(ann.sample)
    labels = list(ann.aux_note) if ann.aux_note is not None else [""] * len(samples)
    for i, samp in enumerate(samples):
        start = float(samp) / fs
        duration = 1.0
        if i + 1 < len(samples):
            duration = max(0.5, float(samples[i + 1] - samp) / fs)
        events.append(
            {
                "start": start,
                "duration": duration,
                "label": str(labels[i]) if i < len(labels) else "arousal",
                "event_type": "arousal",
            }
        )
    return events


def load_annotations(record_path: Path, annotation_format: str = "auto") -> List[dict]:
    stem = record_path.stem
    parent = record_path.parent
    root = parent
    search_roots = [parent, parent.parent, record_path.parents[min(2, len(record_path.parents) - 1)]]
    candidates: List[Path] = []
    for base in search_roots:
        if not base:
            continue
        candidates.extend(
            [
                base / f"{stem}.xml",
                base / f"{stem}-nsrr.xml",
                base / f"{stem}.csv",
                base / f"{stem}.json",
                base / f"{stem}.annotations.json",
            ]
        )
        candidates.extend(base.glob(f"**/{stem}*nsrr*.xml"))
        candidates.extend(base.glob(f"**/{stem}.xml"))
    seen = set()
    for path in candidates:
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        suffix = path.suffix.lower()
        try:
            if suffix == ".xml":
                return parse_nsrr_xml(path)
            if suffix == ".csv":
                return parse_annotation_csv(path)
            if suffix == ".json":
                return parse_annotation_json(path)
        except Exception:
            continue
    if annotation_format in ("cinc", "auto"):
        events = try_load_wfdb_arousal(record_path.with_suffix(""))
        if events:
            return events
    return []


def resample_1d(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if fs_in <= 0 or fs_out <= 0:
        raise ValueError("Sample rates must be positive")
    if abs(fs_in - fs_out) < 1e-6:
        return x.astype(np.float32)
    if resample_poly is None:
        n_out = int(round(len(x) * fs_out / fs_in))
        t_in = np.linspace(0.0, 1.0, num=len(x), endpoint=False)
        t_out = np.linspace(0.0, 1.0, num=max(n_out, 1), endpoint=False)
        return np.interp(t_out, t_in, x).astype(np.float32)
    up = int(round(fs_out))
    down = int(round(fs_in))
    g = math.gcd(up, down)
    return np.asarray(resample_poly(x, up // g, down // g), dtype=np.float32)


def stage_for_interval(annotations: Sequence[dict], t0: float, t1: float) -> int:
    mid = 0.5 * (t0 + t1)
    best = None
    best_overlap = 0.0
    for ev in annotations:
        label = str(ev.get("label") or "")
        etype = str(ev.get("event_type") or "")
        blob = f"{label} {etype}".lower()
        if "stage" not in blob and label.lower() not in STAGE_ALIASES and "wake" not in blob and "rem" not in blob:
            if "n1" not in blob and "n2" not in blob and "n3" not in blob:
                continue
        start = float(ev.get("start") or 0.0)
        end = start + float(ev.get("duration") or 0.0)
        if start <= mid < end or (end <= start and abs(start - t0) < 1e-3):
            return normalize_stage_label(label)
        overlap = max(0.0, min(end, t1) - max(start, t0))
        if overlap > best_overlap:
            best_overlap = overlap
            best = label
    return normalize_stage_label(best) if best else 0


def apnea_for_interval(annotations: Sequence[dict], t0: float, t1: float) -> bool:
    for ev in annotations:
        if not is_apnea_label(str(ev.get("label") or ""), str(ev.get("event_type") or "")):
            continue
        start = float(ev.get("start") or 0.0)
        end = start + float(ev.get("duration") or 0.0)
        if min(end, t1) - max(start, t0) > 0:
            return True
    return False


def map_to_modalities(
    rec: Recording,
    table: ChannelTable,
) -> Tuple[Dict[str, np.ndarray], List[str], Dict[str, List[Optional[str]]]]:
    available = list(rec.signals.keys())
    mapping, warnings = map_recording_channels(available, table)
    target_fs = float(table.target_fs)
    mapped: Dict[str, np.ndarray] = {}
    # Determine a common length after resampling using the longest mapped channel.
    resampled_slots: Dict[str, List[np.ndarray]] = {m: [] for m in MODALITY_ORDER}
    for mod in MODALITY_ORDER:
        n = table.n_channels(mod)
        for i in range(n):
            src = mapping[mod][i] if i < len(mapping[mod]) else None
            if src is None or src not in rec.signals:
                resampled_slots[mod].append(None)  # type: ignore[arg-type]
                continue
            fs_in = float(rec.channel_fs.get(src, target_fs))
            resampled_slots[mod].append(resample_1d(rec.signals[src], fs_in, target_fs))

    lengths = [int(x.shape[0]) for slots in resampled_slots.values() for x in slots if x is not None]
    n_time = min(lengths) if lengths else 0
    for mod in MODALITY_ORDER:
        n = table.n_channels(mod)
        rows = []
        for i in range(n):
            arr = resampled_slots[mod][i] if i < len(resampled_slots[mod]) else None
            if arr is None:
                rows.append(np.zeros(n_time, dtype=np.float32))
            else:
                rows.append(arr[:n_time].astype(np.float32))
        mapped[mod] = np.stack(rows, axis=0) if rows else np.zeros((0, n_time), dtype=np.float32)
    return mapped, warnings, mapping


def epoch_recording(
    rec: Recording,
    table: ChannelTable,
) -> Tuple[List[dict], List[str]]:
    mapped, warnings, _mapping = map_to_modalities(rec, table)
    clip_len = int(table.target_fs * table.clip_seconds)
    n_time = next(iter(mapped.values())).shape[1] if mapped else 0
    n_epochs = n_time // clip_len
    if n_epochs <= 0:
        warnings.append(f"{rec.recording_id}: not enough samples for one {table.clip_seconds}s epoch")
        return [], warnings
    epochs: List[dict] = []
    for i in range(n_epochs):
        sl = slice(i * clip_len, (i + 1) * clip_len)
        stacked = np.concatenate([mapped[m][:, sl] for m in MODALITY_ORDER], axis=0).astype(np.float32)
        t0 = i * table.clip_seconds
        t1 = t0 + table.clip_seconds
        stage_id = stage_for_interval(rec.annotations, t0, t1)
        apnea = apnea_for_interval(rec.annotations, t0, t1)
        missing = [
            m
            for m in MODALITY_ORDER
            if np.allclose(mapped[m][:, sl], 0.0)
        ]
        epochs.append(
            {
                "data": stacked,
                "stage_id": int(stage_id),
                "stage": SLEEP_STAGES[int(stage_id)] if 0 <= stage_id < len(SLEEP_STAGES) else "Wake",
                "apnea": int(apnea),
                "epoch_index": i,
                "participant_id": rec.participant_id,
                "recording_id": rec.recording_id,
                "night_id": rec.recording_id,
                "missing_modalities": missing,
            }
        )
    return epochs, warnings


def discover_recordings(input_dir: str | Path, dataset: str) -> List[Path]:
    input_dir = Path(input_dir)
    if not input_dir.exists():
        return []
    found: List[Path] = []
    if dataset in ("cinc2018", "cinc"):
        found.extend(sorted(input_dir.rglob("*.mat")))
        found.extend(sorted(input_dir.rglob("*.edf")))
        found.extend(sorted(input_dir.rglob("*.EDF")))
        found.extend(sorted(input_dir.rglob("*.npz")))
    else:
        found.extend(sorted(input_dir.rglob("*.edf")))
        found.extend(sorted(input_dir.rglob("*.EDF")))
        found.extend(sorted(input_dir.rglob("*.npz")))
    # Prefer unique stems (mat over edf over npz if duplicates).
    by_stem: Dict[str, Path] = {}
    rank = {".mat": 0, ".edf": 1, ".EDF": 1, ".npz": 2}
    for path in found:
        stem = path.stem
        prev = by_stem.get(stem)
        if prev is None or rank.get(path.suffix, 9) < rank.get(prev.suffix, 9):
            by_stem[stem] = path
    return [by_stem[k] for k in sorted(by_stem)]


def load_recording(path: Path, table: ChannelTable) -> Recording:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        rec = read_npz_recording(path)
    elif suffix == ".mat":
        rec = read_cinc_mat(path)
    else:
        rec = read_edf(path)
    if not rec.annotations:
        rec.annotations = load_annotations(path, table.annotation_format)
    if not rec.participant_id:
        rec.participant_id = rec.recording_id
    return rec


def _channel_slices(channels: Dict[str, int]) -> Dict[str, List[int]]:
    start = 0
    slices: Dict[str, List[int]] = {}
    for mod in MODALITY_ORDER:
        n = channels[mod]
        slices[mod] = [start, start + n]
        start += n
    return slices


def export_recordings(
    recordings: Sequence[Recording],
    output_dir: str | Path,
    table: ChannelTable,
    seed: int = 42,
    split_fractions: Optional[Dict[str, float]] = None,
    dry_run: bool = False,
    dataset_name: Optional[str] = None,
) -> dict:
    """Epoch recordings and write index.json + .npy (or summarize if dry_run)."""
    output_dir = Path(output_dir)
    all_epochs: List[dict] = []
    warnings: List[str] = []
    for rec in recordings:
        epochs, warn = epoch_recording(rec, table)
        warnings.extend(warn)
        all_epochs.extend(epochs)

    pids = [e["participant_id"] for e in all_epochs]
    split_map: Dict[str, str] = {}
    if pids:
        split_map = assign_participant_splits(pids, fractions=split_fractions, seed=seed)

    splits: Dict[str, list] = {s: [] for s in ("pretrain", "valid", "train", "test")}
    channels = dict(table.target_channels)
    clip_length = int(table.target_fs * table.clip_seconds)

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    counters = {s: 0 for s in splits}
    for epoch in all_epochs:
        split = split_map.get(epoch["participant_id"], "pretrain")
        fname = f"{split}_{counters[split]:06d}.npy"
        counters[split] += 1
        entry = {
            "path": fname,
            "stage": epoch["stage"],
            "stage_id": epoch["stage_id"],
            "apnea": epoch["apnea"],
            "participant_id": epoch["participant_id"],
            "recording_id": epoch.get("recording_id"),
            "night_id": epoch.get("night_id"),
            "epoch_index": epoch.get("epoch_index", 0),
        }
        missing = epoch.get("missing_modalities") or []
        if missing:
            entry["missing_modalities"] = missing
        if not dry_run:
            np.save(output_dir / fname, epoch["data"])
        splits[split].append(entry)

    from sleepfm.data.label_coverage import coverage_from_entries, gate_claimed_metrics

    label_cov = coverage_from_entries(
        all_epochs, dataset=dataset_name or table.dataset
    )
    label_gate = gate_claimed_metrics(label_cov)
    meta = {
        "channels": channels,
        "clip_length": clip_length,
        "channel_order": list(MODALITY_ORDER),
        "channel_slices": _channel_slices(channels),
        "synthetic": False,
        "dataset": dataset_name or table.dataset,
        "sample_rate": table.target_fs,
        "clip_seconds": table.clip_seconds,
        "num_participants": len(set(pids)),
        "participant_level_splits": True,
        "missing_channels": document_missing(table),
        "label_coverage": label_cov.to_dict(),
        "label_gate": label_gate.to_dict(),
    }
    if not label_cov.has_aasm_staging or not label_cov.has_respiratory_events:
        warnings.append(
            "LABEL GATE: incomplete AASM/respiratory labels — "
            "staging/SDB metrics will be claim-blocked at eval "
            f"(staging={label_gate.claim_staging}, apnea={label_gate.claim_apnea}). "
            + "; ".join(label_cov.notes[:2])
        )
    summary = {
        "n_recordings": len(recordings),
        "n_epochs": len(all_epochs),
        "n_participants": len(set(pids)),
        "split_counts": {k: len(v) for k, v in splits.items()},
        "warnings": warnings[:50],
        "n_warnings": len(warnings),
        "missing_channels": meta["missing_channels"],
        "label_coverage": meta["label_coverage"],
        "label_gate": meta["label_gate"],
        "dry_run": dry_run,
        "output_dir": str(output_dir),
    }
    if not dry_run:
        with open(output_dir / "index.json", "w", encoding="utf-8") as f:
            json.dump({"meta": meta, "splits": splits}, f, indent=2)
        summary["index"] = str(output_dir / "index.json")
    return summary


def export_dataset(
    input_dir: str | Path,
    output_dir: str | Path,
    dataset: str = "cinc2018",
    channel_config: Optional[str | Path] = None,
    seed: int = 42,
    dry_run: bool = False,
    max_recordings: Optional[int] = None,
    split_fractions: Optional[Dict[str, float]] = None,
) -> dict:
    table = load_channel_table(channel_config or dataset)
    paths = discover_recordings(input_dir, dataset)
    if max_recordings is not None:
        paths = paths[: max(0, int(max_recordings))]
    recordings: List[Recording] = []
    load_errors: List[str] = []
    for path in paths:
        try:
            recordings.append(load_recording(path, table))
        except Exception as exc:
            load_errors.append(f"{path}: {exc}")
    summary = export_recordings(
        recordings,
        output_dir,
        table,
        seed=seed,
        split_fractions=split_fractions,
        dry_run=dry_run,
        dataset_name=dataset,
    )
    summary["n_discovered"] = len(paths)
    summary["load_errors"] = load_errors[:20]
    summary["n_load_errors"] = len(load_errors)
    summary["input_dir"] = str(input_dir)
    summary["dataset"] = dataset
    summary["channel_table"] = table.dataset
    summary["access"] = table.access
    summary["notes"] = table.notes
    if not paths:
        summary["hint"] = (
            f"No EDF/MAT/NPZ files under {input_dir}. "
            "Download CinC 2018 (PhysioNet account) or SHHS/MESA (NSRR DUA), "
            "or run the fixture exporter in tests."
        )
    return summary


def make_fixture_recording(
    recording_id: str,
    participant_id: str,
    fs: int = 256,
    duration_sec: float = 90.0,
    channels: Optional[Dict[str, List[str]]] = None,
    seed: int = 0,
) -> Recording:
    """Tiny in-memory PSG used by dry-run / unit tests (no terabytes required)."""
    rng = np.random.default_rng(seed)
    channels = channels or {
        "bas": ["F3-M2", "C3-M2", "Chin1-Chin2"],
        "ecg": ["ECG"],
        "respiratory": ["ABD", "Chest", "Airflow"],
    }
    n = int(fs * duration_sec)
    t = np.arange(n) / fs
    signals: Dict[str, np.ndarray] = {}
    channel_fs: Dict[str, float] = {}
    for group, names in channels.items():
        for i, name in enumerate(names):
            freq = {"bas": 4.0, "ecg": 1.2, "respiratory": 0.25}[group]
            sig = np.sin(2 * np.pi * freq * (1 + 0.05 * i) * t) + 0.05 * rng.standard_normal(n)
            signals[name] = sig.astype(np.float32)
            channel_fs[name] = float(fs)
    annotations = []
    for i, stage in enumerate((0, 2, 4)):
        annotations.append(
            {
                "start": i * 30.0,
                "duration": 30.0,
                "label": f"Sleep stage {['W', '1', '2', '3', 'R'][stage]}",
                "event_type": "Stages|Stage",
            }
        )
    annotations.append(
        {
            "start": 45.0,
            "duration": 12.0,
            "label": "Obstructive apnea",
            "event_type": "Respiratory|Respiratory Event",
        }
    )
    return Recording(
        recording_id=recording_id,
        participant_id=participant_id,
        signals=signals,
        channel_fs=channel_fs,
        annotations=annotations,
    )
