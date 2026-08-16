"""EDF/NPZ export pipeline: fixtures, dry-run, participant splits, schema."""

from pathlib import Path

import numpy as np
import yaml

from sleepfm.data.channel_map import ChannelTable, load_channel_table, map_recording_channels
from sleepfm.data.edf_export import (
    export_recordings,
    make_fixture_recording,
    parse_nsrr_xml,
    read_edf_basic,
    write_edf_basic,
)
from sleepfm.data.splits import assign_participant_splits, split_overlap
from sleepfm.data.validate import validate_dataset


def _small_table() -> ChannelTable:
    return ChannelTable(
        dataset="fixture",
        target_fs=32,
        clip_seconds=2,
        target_channels={"bas": 2, "ecg": 1, "respiratory": 2},
        slots={
            "bas": [["F3-M2"], ["C3-M2"]],
            "ecg": [["ECG"]],
            "respiratory": [["ABD"], ["Chest"]],
        },
        missing_slots={"ecg": ["second lead padded"]},
    )


def test_load_channel_tables():
    for name in ("cinc2018", "shhs", "mesa"):
        table = load_channel_table(name)
        assert table.target_channels["bas"] == 10
        assert table.target_fs == 256
        assert table.access


def test_channel_mapping_pads_missing():
    table = load_channel_table("cinc2018")
    mapping, warnings = map_recording_channels(
        ["F3-M2", "ECG", "ABD"], table
    )
    assert mapping["bas"][0] == "F3-M2"
    assert mapping["ecg"][0] == "ECG"
    assert mapping["ecg"][1] is None
    assert mapping["bas"][8] is None


def test_write_read_edf_roundtrip(tmp_path):
    fs = 32
    n = fs * 4
    t = np.linspace(0, 1, n, endpoint=False)
    signals = {
        "F3-M2": np.sin(2 * np.pi * 3 * t).astype(np.float32),
        "ECG": np.sin(2 * np.pi * 1.2 * t).astype(np.float32),
    }
    path = write_edf_basic(tmp_path / "mini.edf", signals, fs=fs, patient="P001")
    rec = read_edf_basic(path)
    assert "F3-M2" in rec.signals
    assert rec.signals["F3-M2"].shape[0] >= fs * 3
    assert abs(rec.channel_fs["F3-M2"] - fs) < 1e-3


def test_nsrr_xml_parser(tmp_path):
    xml = """<?xml version="1.0"?>
    <PSGAnnotation>
      <ScoredEvents>
        <ScoredEvent>
          <EventType>Stages|Stage</EventType>
          <EventConcept>Wake|0</EventConcept>
          <Start>0.0</Start>
          <Duration>30.0</Duration>
        </ScoredEvent>
        <ScoredEvent>
          <EventType>Respiratory|Respiratory Events</EventType>
          <EventConcept>Obstructive apnea|Obstructive Apnea</EventConcept>
          <Start>10.0</Start>
          <Duration>12.0</Duration>
        </ScoredEvent>
      </ScoredEvents>
    </PSGAnnotation>
    """
    path = tmp_path / "rec-nsrr.xml"
    path.write_text(xml, encoding="utf-8")
    events = parse_nsrr_xml(path)
    assert len(events) == 2
    assert events[0]["start"] == 0.0


def test_export_fixture_schema_and_no_leak(tmp_path):
    table = _small_table()
    recs = [
        make_fixture_recording(
            f"rec{i}",
            f"S{i:03d}",
            fs=32,
            duration_sec=6.0,
            channels={"bas": ["F3-M2", "C3-M2"], "ecg": ["ECG"], "respiratory": ["ABD", "Chest"]},
            seed=i,
        )
        for i in range(4)
    ]
    out = tmp_path / "export"
    summary = export_recordings(recs, out, table, seed=0, dry_run=False, dataset_name="fixture")
    assert summary["n_participants"] == 4
    assert summary["n_epochs"] > 0
    ok, msgs = validate_dataset(out, strict_participants=True)
    assert ok, msgs
    has, overlap = split_overlap(out, "pretrain", "train", by="participant_id")
    assert not has, overlap
    has_p, overlap_p = split_overlap(out, "pretrain", "test", by="path")
    assert not has_p, overlap_p


def test_export_dry_run_no_files(tmp_path):
    table = _small_table()
    recs = [
        make_fixture_recording(f"rec{i}", f"S{i:03d}", fs=32, duration_sec=6.0, seed=i)
        for i in range(4)
    ]
    out = tmp_path / "dry"
    summary = export_recordings(recs, out, table, dry_run=True)
    assert summary["dry_run"] is True
    assert not (out / "index.json").exists()
    assert summary["n_epochs"] > 0


def test_assign_participant_splits_disjoint():
    mapping = assign_participant_splits([f"P{i}" for i in range(20)], seed=1)
    by_split = {}
    for pid, split in mapping.items():
        by_split.setdefault(split, set()).add(pid)
    assert set(by_split) >= {"pretrain", "valid", "train", "test"}
    ids = list(by_split.values())
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            assert not (ids[i] & ids[j])


def test_channel_yaml_on_disk():
    path = Path("configs/channels/cinc2018.yaml")
    if not path.is_file():
        path = Path(__file__).resolve().parents[1] / "configs" / "channels" / "cinc2018.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert raw["target_channels"]["respiratory"] == 7
