# SleepFM on-disk data schema

## Layout

```
data/<dataset_name>/
  index.json
  pretrain_000000.npy
  train_000012.npy
  ...
```

## `index.json`

```json
{
  "meta": {
    "channels": {"bas": 10, "ecg": 2, "respiratory": 7},
    "clip_length": 7680,
    "channel_order": ["bas", "ecg", "respiratory"],
    "channel_slices": {
      "bas": [0, 10],
      "ecg": [10, 12],
      "respiratory": [12, 19]
    },
    "synthetic": true,
    "num_participants": 24,
    "participant_level_splits": true
  },
  "splits": {
    "pretrain": [{"path": "pretrain_000000.npy", "stage_id": 2, "apnea": 0, "participant_id": "P0001"}],
    "valid": [],
    "train": [],
    "test": []
  }
}
```

### Required split names (paper)

| Split | Use |
|-------|-----|
| `pretrain` | Contrastive pretraining only |
| `valid` | Early stopping (pretrain) and optional L2 `C` tuning (downstream) |
| `train` | Logistic regression on frozen embeddings |
| `test` | Held-out downstream metrics |

**Cohort rule:** No `path` or `participant_id` may appear in both `pretrain` and `train`/`test`.

## `.npy` epoch files

- Shape: `(C, T)` float32
- `C` = sum of modality channel counts
- `T` = `meta.clip_length` (e.g. 256 Hz × 30 s → 7680)

Row order matches `meta.channel_slices`.

## Optional index fields (SleepFM-Unify export)

Required keys are unchanged. Exporters may also write:

| Field | Meaning |
|-------|---------|
| `epoch_index` | Order within a night (0-based) |
| `night_id` / `recording_id` | Night / file id (defaults to `participant_id`) |
| `missing_modalities` | e.g. `["ecg"]` — loader zero-fills that group |

`meta` may include `sample_rate`, `clip_seconds`, `dataset`, `missing_channels` (documented padded leads), plus optional `label_coverage` / `label_gate` (CinC incomplete AASM/respiratory detection). Do not remove `channels` / `channel_slices` / split names.

## Validation

```powershell
python scripts/validate_data.py --data-dir data/synthetic --strict-participants
```

Programmatic: `sleepfm.data.validate.validate_dataset(data_dir)`.
