import json
from pathlib import Path

import numpy as np
import torch
from astropy.io import fits
from benchopt import BaseDataset
from torch.utils.data import DataLoader, Dataset as TorchDataset
from toolsbench.utils.deepinv_imager import (
            DeepinvDirtyImager,
            DirtyImagerConfig,
        )


class RadioStreamDataset(TorchDataset):
    """Dataset that loads cached image tensors and per-sample physics specs."""

    def __init__(self, records):
        self.records = list(records)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        with fits.open(record["image_path"], memmap=False) as hdul:
            image_np = hdul[0].data.astype(np.float32)
            image = torch.from_numpy(image_np)
        return {
            "image": image,
            "image_path": record["image_path"],
            "physics_spec": dict(record["physics_spec"]),
            "measurements_path": record["measurements_path"],
        }


class Dataset(BaseDataset):
    """Synthetic stream dataset backed by on-disk images."""

    name = "single_image_blur_stream"
    requirements = [
        "pip::torch",
        "numpy",
    ]

    parameters = {
        "image_size": [256],
        "noise_level": [0.01],
        "stream_length": [64],
        "rate_hz": [0.0],  # 0.0 -> as fast as possible
        "queue_capacity": [4],
        "drop_policy": ["block"],
        "seed": [42],
    }

    def get_data(self):
        device = torch.device("cpu")
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        records = self._build_stream_records(
            data_dir=self._stream_data_dir(),
            device=device,
        )
        stream_dataset = RadioStreamDataset(records)
        stream_dataloader = DataLoader(
            stream_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=0,
        )
        if len(stream_dataset) == 0:
            raise ValueError("stream_length must be >= 1 to create a valid stream.")

        first_sample = stream_dataset[0]
        ground_truth = first_sample["image"].clone()
        physics_spec = dict(first_sample["physics_spec"])

        stream_spec = {
            "rate_hz": (None if self.rate_hz <= 0.0 else float(self.rate_hz)),
            "max_packets": len(stream_dataset),
            "queue_capacity": int(self.queue_capacity),
            "drop_policy": self.drop_policy,
            "include_ground_truth": True,
        }

        return dict(
            ground_truth=ground_truth,
            stream_dataloader=stream_dataloader,
            physics_spec=physics_spec,
            stream_spec=stream_spec,
            min_pixel=0.0,
            max_pixel=1.0,
        )

    def _stream_data_dir(self):
        root = Path(__file__).resolve().parents[1] / "data" / "radio_interferometry"
        return root / (
            f"size_{int(self.image_size)}_seed_{int(self.seed)}"
            f"_blur_{float(self.blur_sigma):.3f}_noise_{float(self.noise_level):.5f}"
        )

    def _build_stream_records(self, data_dir, device):
        data_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for sample_id in range(int(self.stream_length)):
            image_path = data_dir / f"image_{sample_id:06d}.fits"
            ms_path = data_dir / f"ms_{sample_id:06d}.ms"
            metadata_path = data_dir / f"metadata_{sample_id:06d}.json"
            if metadata_path.exists():
                with metadata_path.open("r", encoding="utf-8") as f:
                    metadata = json.load(f)
                imaging_cellsize = float(metadata["imaging_cellsize"])

            imager_config = DirtyImagerConfig(
                imaging_npixel=self.image_size,
                imaging_cellsize=imaging_cellsize,
                combine_across_frequencies=False,
            )

            imager = DeepinvDirtyImager(imager_config, device=device)

            # create_deepinv_physics loads the MS and builds the operator
            samples_locs, measurements, weights = imager.read_ms(
                visibility_path=str(ms_path),
                visibility_format="MS",
                visibility_column="DATA",
            )
            records.append(
                {
                    "image_path": str(image_path),
                    "physics_spec": {
                        "samples_locs": samples_locs,
                        "weights": weights,
                        "noise_level": float(self.noise_level),
                        "seed": int(self.seed),
                    },
                    "measurements_path": str(ms_path),
                }
            )
        return records
