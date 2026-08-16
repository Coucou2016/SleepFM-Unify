"""End-to-end paper experiment suite (validate → pretrain → downstream → extras).

Writes JSON under ``outputs/paper_suite/``. Use ``--demo`` for synthetic CI runs
that do not require CinC/SHHS on disk.

Temporal night head: pass ``--temporal-checkpoint`` or ``--train-temporal``
(uses ``configs/unify_temporal.yaml``). ``--demo`` stays fast — temporal
pretrain is opt-in only (not part of the default dual LOO+Unify path).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _save(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def _run_validate(data_dir: str, strict: bool = True) -> Dict:
    from sleepfm.data.validate import validate_dataset

    ok, messages = validate_dataset(data_dir, strict_participants=strict)
    return {"ok": ok, "messages": messages}


def _ensure_demo_data(cfg: dict, data_dir: Path) -> None:
    from sleepfm.data.synthetic import write_synthetic_dataset

    if (data_dir / "index.json").is_file():
        return
    demo = cfg["demo"]
    clip_length = int(demo["sample_rate"] * demo["clip_seconds"])
    write_synthetic_dataset(
        data_dir,
        cfg["channels"],
        clip_length,
        {
            "pretrain": demo.get("num_pretrain", 64),
            "valid": demo["num_val"],
            "train": demo.get("num_train", 48),
            "test": demo["num_test"],
        },
        seed=cfg["seed"],
        apnea_rate=demo.get("apnea_rate", 0.2),
        num_participants=demo.get("num_participants"),
        epochs_per_participant=demo.get("epochs_per_participant"),
    )


def _pretrain(
    cfg: dict,
    data_dir: str,
    output_dir: Path,
    *,
    demo: bool,
    unify: bool,
    config_path: str,
    temporal: bool = False,
) -> str:
    from sleepfm.data.dataset import SleepEpochDataset, collate_multimodal
    from sleepfm.data.night_dataset import NightSequenceDataset, collate_night
    from sleepfm.models.sleepfm import MultiModalSleepFM
    from sleepfm.models.temporal import NightTemporalEncoder
    from sleepfm.training.trainer import PretrainTrainer
    from sleepfm.utils.config import load_config
    from sleepfm.utils.seed import set_seed

    # Reload so unify.yaml / default.yaml / unify_temporal.yaml differ cleanly
    local = load_config(config_path)
    set_seed(local["seed"])
    demo_cfg = local["demo"] if demo else {}
    batch_size = demo_cfg.get("batch_size", local["batch_size"])
    epochs = demo_cfg.get("epochs", local["epochs"])
    unify_cfg = dict(local.get("unify") or {})
    if unify:
        unify_cfg["enabled"] = True
    temporal_cfg = dict(local.get("temporal") or {})
    if temporal:
        temporal_cfg["enabled"] = True
        unify_cfg["enabled"] = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if temporal_cfg.get("enabled"):
        window = int(temporal_cfg.get("window", 8))
        stride = temporal_cfg.get("stride")
        # Demo keeps sequences short so CPU CI stays fast.
        if demo:
            window = min(window, int(demo_cfg.get("epochs_per_participant", window) or window))
            window = max(2, window)
        train_ds = NightSequenceDataset(
            data_dir, split="pretrain", window=window, stride=stride
        )
        val_ds = NightSequenceDataset(
            data_dir, split="valid", window=window, stride=stride
        )
        collate_fn = collate_night
    else:
        window = 8
        train_ds = SleepEpochDataset(data_dir, split="pretrain")
        val_ds = SleepEpochDataset(data_dir, split="valid")
        collate_fn = collate_multimodal

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    unify_on = bool(unify_cfg.get("enabled"))
    model = MultiModalSleepFM(
        channels=local["channels"],
        embedding_dim=local["embedding_dim"],
        temperature_init=local["temperature_init"],
        unify=unify_on,
        shared_dim=unify_cfg.get("shared_dim"),
        private_dim=unify_cfg.get("private_dim"),
        downstream_space=unify_cfg.get("downstream_space", "concat"),
    )
    temporal_encoder = None
    if temporal_cfg.get("enabled"):
        d_model = int(unify_cfg.get("shared_dim", 256) if unify_on else local["embedding_dim"])
        temporal_encoder = NightTemporalEncoder(
            d_model=d_model,
            n_layers=int(temporal_cfg.get("n_layers", 2)),
            n_heads=int(temporal_cfg.get("n_heads", 4)),
            kind=str(temporal_cfg.get("model", "gru")),
            window=window,
        )
    loss_weights = dict(unify_cfg.get("loss_weights") or {}) if unify_on else None
    if temporal_encoder is not None:
        loss_weights = dict(loss_weights or {})
        if float(loss_weights.get("temporal", 0.0)) <= 0.0:
            loss_weights["temporal"] = float(temporal_cfg.get("loss_weight", 0.2))

    output_dir.mkdir(parents=True, exist_ok=True)
    trainer = PretrainTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=local["lr"],
        momentum=local["momentum"],
        weight_decay=local["weight_decay"],
        lr_step_period=local["lr_step_period"],
        lr_gamma=local["lr_gamma"],
        output_dir=str(output_dir),
        contrastive_mode=local["contrastive_mode"],
        early_stopping_patience=local.get("early_stopping_patience"),
        loss_weights=loss_weights,
        modality_dropout=float(unify_cfg.get("modality_dropout", 0.0) if unify_on else 0.0),
        temporal_encoder=temporal_encoder,
        temporal_mask_prob=float(temporal_cfg.get("mask_prob", 0.15)),
        use_mixed_loss=unify_on or temporal_encoder is not None,
    )
    trainer.fit(epochs=epochs)
    return str(output_dir / "best.pt")


def _check_channels(checkpoint: str, data_dir: str, allow_mismatch: bool) -> Dict:
    from sleepfm.models.channel_meta import assert_channels_compatible
    from sleepfm.models.sleepfm import MultiModalSleepFM

    model = MultiModalSleepFM.from_checkpoint(checkpoint, device="cpu")
    report = assert_channels_compatible(
        model.channels,
        data_dir=data_dir,
        allow_mismatch=allow_mismatch,
        context=f"checkpoint={checkpoint}",
    )
    return {
        "ok": report.ok,
        "model_channels": report.model_channels,
        "data_channels": report.data_channels,
        "messages": report.messages,
        "allow_mismatch": allow_mismatch,
    }


def _label_gate(data_dir: str) -> Dict:
    from sleepfm.data.label_coverage import compute_label_coverage, gate_claimed_metrics

    cov = compute_label_coverage(data_dir)
    gate = gate_claimed_metrics(cov)
    return {"coverage": cov.to_dict(), "gate": gate.to_dict()}


def _downstream(
    cfg: dict,
    data_dir: str,
    checkpoint: str,
    batch_size: int,
    *,
    allow_channel_mismatch: bool = False,
    apply_label_gate: bool = True,
) -> Dict:
    from sleepfm.data.dataset import SleepEpochDataset
    from sleepfm.data.label_coverage import (
        apply_metric_gate,
        compute_label_coverage,
        gate_claimed_metrics,
    )
    from sleepfm.eval.downstream import (
        build_embedding_matrix,
        can_train_binary,
        evaluate_apnea,
        evaluate_sleep_staging,
        train_logistic_regression,
        tune_lr_c,
    )
    from sleepfm.models.channel_meta import assert_channels_compatible
    from sleepfm.models.sleepfm import MultiModalSleepFM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiModalSleepFM.from_checkpoint(checkpoint, device=str(device))
    assert_channels_compatible(
        model.channels,
        data_dir=data_dir,
        allow_mismatch=allow_channel_mismatch,
        context="downstream",
    )
    model.to(device)
    ds_cfg = cfg["downstream"]
    train_ds = SleepEpochDataset(data_dir, split=ds_cfg.get("train_split", "train"), return_labels=True)
    test_ds = SleepEpochDataset(data_dir, split=ds_cfg.get("test_split", "test"), return_labels=True)
    X_tr, y_tr = build_embedding_matrix(model, train_ds, device, batch_size)
    X_te, y_te = build_embedding_matrix(model, test_ds, device, batch_size)
    lr_cfg = dict(ds_cfg)
    if ds_cfg.get("tune_c_on_valid"):
        try:
            valid_ds = SleepEpochDataset(
                data_dir, split=ds_cfg.get("valid_split", "valid"), return_labels=True
            )
            X_va, y_va = build_embedding_matrix(model, valid_ds, device, batch_size)
            lr_cfg["C"] = tune_lr_c(X_va, y_va["stage_id"], {**ds_cfg, "task": "staging"})
        except KeyError:
            pass
    out: Dict[str, Any] = {
        "staging": evaluate_sleep_staging(
            train_logistic_regression(X_tr, y_tr["stage_id"], lr_cfg), X_te, y_te["stage_id"]
        )
    }
    if can_train_binary(y_tr["apnea"]):
        out["apnea"] = evaluate_apnea(
            train_logistic_regression(X_tr, y_tr["apnea"], lr_cfg), X_te, y_te["apnea"]
        )
    else:
        out["apnea"] = {"skipped": True, "reason": "need >=2 apnea classes in train"}
    if apply_label_gate:
        gate = gate_claimed_metrics(compute_label_coverage(data_dir))
        out = apply_metric_gate(out, gate)
    return out


def _retrieval(
    checkpoint: str,
    data_dir: str,
    split: str,
    max_gallery: Optional[int],
    k: int = 10,
    *,
    gallery_seed: int = 0,
    gallery_mode: str = "rng",
) -> Dict:
    from sleepfm.data.dataset import SleepEpochDataset, collate_multimodal
    from sleepfm.eval.retrieval import limit_gallery, modality_retrieval_metrics, random_recall_baseline
    from sleepfm.models.sleepfm import MultiModalSleepFM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiModalSleepFM.from_checkpoint(checkpoint, device=str(device))
    model.to(device).eval()
    ds = SleepEpochDataset(data_dir, split=split)
    loader = DataLoader(ds, batch_size=32, collate_fn=collate_multimodal)
    all_emb = {m: [] for m in model.MODALITY_ORDER}
    n_got = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            space = "shared" if getattr(model, "unify", False) else "downstream"
            z = model.encode(batch, space=space)
            for m, t in z.items():
                all_emb[m].append(t.cpu())
            if z:
                n_got += next(iter(z.values())).size(0)
            # Collect more than the cap so RNG subsample is not a prefix of loader order.
            collect_target = None if max_gallery is None else max(max_gallery * 4, max_gallery)
            if collect_target is not None and n_got >= collect_target:
                break
    embeddings = {m: torch.cat(c, dim=0) for m, c in all_emb.items() if c}
    embeddings = limit_gallery(
        embeddings,
        max_gallery=max_gallery,
        seed=gallery_seed,
        mode=gallery_mode,
    )
    metrics = modality_retrieval_metrics(embeddings, k=k)
    n = next(iter(embeddings.values())).size(0)
    return {
        "n": n,
        "k": k,
        "random_baseline": random_recall_baseline(n, k=k),
        "metrics": metrics,
        "gallery_mode": gallery_mode,
        "gallery_seed": gallery_seed,
    }


def _ablation(cfg: dict, checkpoint: str, data_dir: str, batch_size: int) -> Dict:
    from sleepfm.eval.experiments import modality_ablation_table
    from sleepfm.models.sleepfm import MultiModalSleepFM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiModalSleepFM.from_checkpoint(checkpoint, device=str(device))
    model.to(device)
    return modality_ablation_table(
        model, data_dir, device, cfg["downstream"], batch_size=batch_size
    )


def _fewshot(cfg: dict, checkpoint: str, data_dir: str, ks, batch_size: int) -> Dict:
    from sleepfm.eval.experiments import fewshot_curve
    from sleepfm.models.sleepfm import MultiModalSleepFM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiModalSleepFM.from_checkpoint(checkpoint, device=str(device))
    model.to(device)
    return fewshot_curve(
        model,
        data_dir,
        device,
        cfg["downstream"],
        ks=ks,
        seed=cfg["seed"],
        batch_size=batch_size,
        n_repeats=2 if len(ks) <= 3 else 3,
    )


def _night(
    checkpoint: str,
    data_dir: str,
    batch_size: int,
    *,
    allow_channel_mismatch: bool = False,
    apply_label_gate: bool = True,
) -> Dict:
    from sleepfm.data.label_coverage import (
        apply_metric_gate,
        compute_label_coverage,
        gate_claimed_metrics,
    )
    from sleepfm.eval.night import epoch_sequence_kappa, night_eval_pack, probe_night_tasks
    from sleepfm.models.channel_meta import assert_channels_compatible
    from sleepfm.models.sleepfm import MultiModalSleepFM
    from sleepfm.models.temporal import NightTemporalEncoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiModalSleepFM.from_checkpoint(checkpoint, device=str(device))
    assert_channels_compatible(
        model.channels,
        data_dir=data_dir,
        allow_mismatch=allow_channel_mismatch,
        context="night",
    )
    model.to(device)
    temporal = NightTemporalEncoder.from_checkpoint(checkpoint, device=str(device))
    tr = night_eval_pack(model, data_dir, "train", device, batch_size=batch_size, temporal_encoder=temporal)
    te = night_eval_pack(model, data_dir, "test", device, batch_size=batch_size, temporal_encoder=temporal)
    metrics = probe_night_tasks(tr["X"], tr["summaries"], te["X"], te["summaries"])
    metrics["temporal_head"] = "loaded" if temporal is not None else "mean_pool_fallback"
    if tr["y_stage"].size and te["y_stage"].size:
        metrics["staging_epoch_kappa"] = epoch_sequence_kappa(
            tr["X_epochs"], tr["y_stage"], te["X_epochs"], te["y_stage"]
        )
    # Drop large night key lists from suite JSON
    metrics["train_n_nights"] = len(tr["keys"])
    metrics["test_n_nights"] = len(te["keys"])
    if apply_label_gate:
        gate = gate_claimed_metrics(compute_label_coverage(data_dir))
        metrics = apply_metric_gate(
            metrics,
            gate,
            staging_keys=("staging_epoch_kappa",),
            apnea_keys=(),
            night_ahi_keys=("ahi_bin_auroc", "ahi_bin", "ahi"),
        )
        # probe_night_tasks may use different key names — gate common summary probes
        if not gate.claim_night_ahi:
            for k in list(metrics.keys()):
                if "ahi" in k.lower() and k not in ("label_gate",):
                    if isinstance(metrics[k], dict) and metrics[k].get("skipped"):
                        continue
                    metrics[k] = {
                        "skipped": True,
                        "reason": "insufficient respiratory/apnea label coverage",
                        "prior_value_removed": True,
                    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="SleepFM paper experiment suite")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--unify-config", type=str, default="configs/unify.yaml")
    parser.add_argument(
        "--temporal-config",
        type=str,
        default="configs/unify_temporal.yaml",
        help="Config used when --train-temporal is set",
    )
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs/paper_suite")
    parser.add_argument("--demo", action="store_true", help="Synthetic demo for CI (keeps temporal opt-in)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Skip LOO pretrain; use this ckpt")
    parser.add_argument("--unify-checkpoint", type=str, default=None, help="Skip Unify pretrain")
    parser.add_argument(
        "--temporal-checkpoint",
        type=str,
        default=None,
        help="Night eval ckpt with temporal_state_dict (skips --train-temporal)",
    )
    parser.add_argument(
        "--train-temporal",
        action="store_true",
        help="Run unify_temporal pretrain so night eval uses a real temporal head",
    )
    parser.add_argument("--skip-unify", action="store_true")
    parser.add_argument("--skip-pretrain", action="store_true")
    parser.add_argument(
        "--allow-channel-mismatch",
        action="store_true",
        help="Override fail-fast when checkpoint channels != data meta (5/1/3 vs 10/2/7)",
    )
    parser.add_argument("--max-gallery", type=int, default=None)
    parser.add_argument(
        "--gallery-seed",
        type=int,
        default=None,
        help="Seed for gallery RNG subsample (default: config seed)",
    )
    parser.add_argument(
        "--gallery-mode",
        type=str,
        choices=["rng", "prefix"],
        default="rng",
        help="Gallery cap mode: rng (default) or prefix (legacy)",
    )
    parser.add_argument("--fewshot-ks", type=str, default="1,2,4")
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    from sleepfm.utils.config import load_config
    from sleepfm.utils.seed import set_seed

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    gallery_seed = args.gallery_seed if args.gallery_seed is not None else int(cfg["seed"])
    demo = bool(args.demo)
    data_dir = Path(args.data_dir or (cfg["data_dir"] if not demo else "data/synthetic"))
    out_root = Path(args.output_dir) / (_now() + ("_demo" if demo else ""))
    out_root.mkdir(parents=True, exist_ok=True)
    batch_size = args.batch_size or (cfg["demo"].get("batch_size", 8) if demo else cfg["batch_size"])
    if demo and args.max_gallery is None:
        args.max_gallery = 64

    results: Dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "demo": demo,
        "data_dir": str(data_dir),
        "steps": {},
    }
    t0 = time.time()

    if demo:
        print(f"[1/9] Ensuring synthetic data at {data_dir}")
        _ensure_demo_data(cfg, data_dir)

    print("[1/9] validate_data")
    results["steps"]["validate"] = _run_validate(str(data_dir), strict=True)
    _save(out_root / "01_validate.json", results["steps"]["validate"])
    if not results["steps"]["validate"]["ok"]:
        print("Validation failed; aborting suite.")
        _save(out_root / "summary.json", results)
        sys.exit(1)

    print("[1b/9] label coverage gate")
    results["steps"]["label_coverage"] = _label_gate(str(data_dir))
    _save(out_root / "01b_label_coverage.json", results["steps"]["label_coverage"])
    for msg in results["steps"]["label_coverage"]["gate"].get("messages", []):
        print(f"  {msg}")

    loo_ckpt = args.checkpoint
    if loo_ckpt is None and not args.skip_pretrain:
        print("[2/9] baseline LOO pretrain")
        loo_dir = out_root / "pretrain_loo"
        loo_ckpt = _pretrain(
            cfg, str(data_dir), loo_dir, demo=demo, unify=False, config_path=args.config
        )
        results["steps"]["pretrain_loo"] = {"checkpoint": loo_ckpt}
    elif loo_ckpt:
        print(f"[2/9] skip LOO pretrain (checkpoint={loo_ckpt})")
        results["steps"]["pretrain_loo"] = {"checkpoint": loo_ckpt, "skipped": True}
    else:
        print("[2/9] skip LOO pretrain (--skip-pretrain without --checkpoint)")
        results["steps"]["pretrain_loo"] = {"skipped": True}

    unify_ckpt = args.unify_checkpoint
    if unify_ckpt is None and not args.skip_unify:
        print("[3/9] unify pretrain")
        unify_dir = out_root / "pretrain_unify"
        unify_ckpt = _pretrain(
            cfg,
            str(data_dir),
            unify_dir,
            demo=demo,
            unify=True,
            config_path=args.unify_config,
        )
        results["steps"]["pretrain_unify"] = {"checkpoint": unify_ckpt}
    elif unify_ckpt:
        print(f"[3/9] skip unify pretrain (checkpoint={unify_ckpt})")
        results["steps"]["pretrain_unify"] = {"checkpoint": unify_ckpt, "skipped": True}
    else:
        print("[3/9] skip unify pretrain")
        results["steps"]["pretrain_unify"] = {"skipped": True}

    temporal_ckpt = args.temporal_checkpoint
    if temporal_ckpt is None and args.train_temporal:
        print("[3b/9] unify_temporal pretrain")
        temporal_dir = out_root / "pretrain_temporal"
        temporal_ckpt = _pretrain(
            cfg,
            str(data_dir),
            temporal_dir,
            demo=demo,
            unify=True,
            temporal=True,
            config_path=args.temporal_config,
        )
        results["steps"]["pretrain_temporal"] = {"checkpoint": temporal_ckpt}
    elif temporal_ckpt:
        print(f"[3b/9] using temporal checkpoint ({temporal_ckpt})")
        results["steps"]["pretrain_temporal"] = {
            "checkpoint": temporal_ckpt,
            "skipped": True,
        }
    else:
        print(
            "[3b/9] skip temporal pretrain "
            "(pass --train-temporal or --temporal-checkpoint for a real night head)"
        )
        results["steps"]["pretrain_temporal"] = {"skipped": True}
        if demo:
            print("  note: --demo stays fast; temporal is opt-in")

    eval_ckpt = unify_ckpt or loo_ckpt
    if not eval_ckpt:
        print("No checkpoint available for downstream steps; stopping after validate/pretrain.")
        results["elapsed_sec"] = time.time() - t0
        _save(out_root / "summary.json", results)
        print(f"Wrote {out_root / 'summary.json'}")
        return

    print("[4/9] channel meta check")
    try:
        results["steps"]["channel_check"] = _check_channels(
            eval_ckpt, str(data_dir), args.allow_channel_mismatch
        )
    except RuntimeError as exc:
        results["steps"]["channel_check"] = {"ok": False, "error": str(exc)}
        _save(out_root / "04_channel_check.json", results["steps"]["channel_check"])
        print(str(exc))
        print("Aborting (use --allow-channel-mismatch to override).")
        results["elapsed_sec"] = time.time() - t0
        _save(out_root / "summary.json", results)
        sys.exit(1)
    _save(out_root / "04_channel_check.json", results["steps"]["channel_check"])

    print("[5/9] downstream staging + apnea (gated)")
    results["steps"]["downstream"] = _downstream(
        cfg,
        str(data_dir),
        eval_ckpt,
        batch_size,
        allow_channel_mismatch=args.allow_channel_mismatch,
    )
    if loo_ckpt and unify_ckpt and loo_ckpt != unify_ckpt:
        results["steps"]["downstream_loo"] = _downstream(
            cfg,
            str(data_dir),
            loo_ckpt,
            batch_size,
            allow_channel_mismatch=args.allow_channel_mismatch,
        )
    _save(out_root / "05_downstream.json", results["steps"]["downstream"])

    print("[6/9] retrieval")
    results["steps"]["retrieval"] = _retrieval(
        eval_ckpt,
        str(data_dir),
        "pretrain",
        args.max_gallery,
        gallery_seed=gallery_seed,
        gallery_mode=args.gallery_mode,
    )
    _save(out_root / "06_retrieval.json", results["steps"]["retrieval"])

    print("[7/9] modality ablation")
    results["steps"]["modality_ablation"] = _ablation(cfg, eval_ckpt, str(data_dir), batch_size)
    _save(out_root / "07_modality_ablation.json", results["steps"]["modality_ablation"])

    print("[8/9] few-shot")
    ks = [int(x) for x in args.fewshot_ks.split(",") if x.strip()]
    results["steps"]["fewshot"] = _fewshot(cfg, eval_ckpt, str(data_dir), ks, batch_size)
    _save(out_root / "08_fewshot.json", results["steps"]["fewshot"])

    night_ckpt = temporal_ckpt or eval_ckpt
    print("[9/9] night eval")
    try:
        results["steps"]["night"] = _night(
            night_ckpt,
            str(data_dir),
            batch_size,
            allow_channel_mismatch=args.allow_channel_mismatch,
        )
    except Exception as exc:  # pragma: no cover - defensive for sparse nights
        results["steps"]["night"] = {"error": str(exc)}
    _save(out_root / "09_night.json", results["steps"]["night"])

    results["elapsed_sec"] = time.time() - t0
    results["finished_at"] = datetime.now(timezone.utc).isoformat()
    results["primary_checkpoint"] = eval_ckpt
    results["night_checkpoint"] = night_ckpt
    _save(out_root / "summary.json", results)
    print(f"PAPER SUITE DONE → {out_root / 'summary.json'} ({results['elapsed_sec']:.1f}s)")


if __name__ == "__main__":
    main()
