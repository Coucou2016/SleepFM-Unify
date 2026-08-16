"""
NATL60 OSSE data loader stub (official Wasabi URLs).

Full pipeline requires NetCDF preprocessing into ``train.npz`` / ``val.npz`` / ``test.npz``
with the same keys as ``OSSEDataset``. See README and:

- Obs: https://s3.eu-central-1.wasabisys.com/melody/NATL/data/gridded_data_swot_wocorr/dataset_nadir_0d_swot.nc
- OI SSH: https://s3.eu-central-1.wasabisys.com/melody/NATL/oi/ssh_NATL60_swot_4nadir.nc
- Ref SSH/SST/u/v: https://s3.eu-central-1.wasabisys.com/melody/NATL/ref/
"""

from __future__ import annotations

from pathlib import Path


def natl60_data_urls() -> dict[str, str]:
    base = "https://s3.eu-central-1.wasabisys.com/melody/NATL"
    return {
        "obs": f"{base}/data/gridded_data_swot_wocorr/dataset_nadir_0d_swot.nc",
        "oi_ssh": f"{base}/oi/ssh_NATL60_swot_4nadir.nc",
        "ref_ssh": f"{base}/ref/NATL60-CJM165_NATL_ssh_y2013.1y.nc",
        "ref_sst": f"{base}/ref/NATL60-CJM165_NATL_sst_y2013.1y.nc",
        "ref_u": f"{base}/ref/NATL60-CJM165_NATL_u_y2013.1y.nc",
        "ref_v": f"{base}/ref/NATL60-CJM165_NATL_v_y2013.1y.nc",
    }


def expected_npz_layout(data_dir: str | Path) -> list[Path]:
    data_dir = Path(data_dir)
    return [data_dir / f"{split}.npz" for split in ("train", "val", "test")]
