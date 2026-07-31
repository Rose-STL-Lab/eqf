from __future__ import annotations

from typing import Any

import torch
from einops import rearrange

from algorithms.wan.pose_utils import (
    select_latent_pose_indices,
)
from utils.geometry_utils import CameraPose


class WanPoseConditioningMixin:
    def _camera_pose_cfg(self) -> Any:
        if not hasattr(self.cfg, "camera_pose_conditioning"):
            raise ValueError(
                f"{type(self).__name__} requires `camera_pose_conditioning` in config."
            )
        return self.cfg.camera_pose_conditioning

    def _pose_conditioning_type(self) -> str:
        return str(self._camera_pose_cfg().type).lower()

    def _pose_latent_hw(self) -> tuple[int, int]:
        if hasattr(self, "lat_h") and hasattr(self, "lat_w"):
            return int(self.lat_h), int(self.lat_w)
        if hasattr(self, "latent_size") and self.latent_size is not None:
            lat_w, lat_h = self.latent_size
            return int(lat_h), int(lat_w)
        raise ValueError(f"{type(self).__name__} could not infer latent H/W.")

    def _pose_temporal_stride(self) -> int:
        if hasattr(self, "vae_stride"):
            return int(self.vae_stride[0])
        if hasattr(self, "temporal_downsampling_factor"):
            return int(self.temporal_downsampling_factor)
        raise ValueError(f"{type(self).__name__} could not infer temporal stride.")

    def _downsample_pose_time(self, pose: torch.Tensor) -> torch.Tensor:
        indices = select_latent_pose_indices(
            num_frames=int(pose.shape[1]),
            stride=self._pose_temporal_stride(),
            downsample_technique=self._camera_pose_cfg().downsample_technique,
        ).to(device=pose.device)
        return pose.index_select(1, indices)

    def _encode_conditions(
        self, conditions: torch.Tensor | None
    ) -> torch.Tensor | None:
        """Convert raw `(B, T, 16)` camera vectors into the latent-grid pose
        feature tensor consumed by `WanModelPose`.

        Steps: rigid-normalize, optional scale-bound, build rays/plucker/encoding
        on the latent H/W grid, and temporally subsample to the VAE latent rate.
        """
        if conditions is None:
            return conditions
        if conditions.ndim != 3 or conditions.shape[-1] != 16:
            raise ValueError(
                "Raw WAN pose conditions must have shape (B, T, 16) containing "
                "4 intrinsics + 12 extrinsics."
            )

        input_dtype = conditions.dtype
        camera_poses = CameraPose.from_vectors(conditions.float())
        pose_cfg = self._camera_pose_cfg()

        normalize_by = str(pose_cfg.normalize_by).lower()
        if normalize_by == "first":
            camera_poses.normalize_by_first()
        elif normalize_by == "mean":
            camera_poses.normalize_by_mean()
        else:
            raise ValueError(
                f"Unsupported camera pose normalization mode: {pose_cfg.normalize_by}"
            )

        bound = getattr(pose_cfg, "bound", None)
        if bound is not None:
            camera_poses.scale_within_bounds(float(bound))

        cond_type = self._pose_conditioning_type()

        lat_h, lat_w = self._pose_latent_hw()
        # RE10K pose-conditioned WAN assumes pose maps are constructed directly on
        # the WAN latent grid. We use latent H/W here instead of image resolution.
        rays = camera_poses.rays_hw(lat_h, lat_w)
        if cond_type == "ray":
            processed = rays.to_tensor(use_plucker=False) # (B, T, H, W, 6)
        elif cond_type == "plucker":
            processed = rays.to_tensor(use_plucker=True) # (B, T, H, W, 6)
        elif cond_type == "ray_encoding":
            processed = rays.to_pos_encoding()[0] # (B, T, H, W, 180); 180-dim encoding with freq_origin=15, freq_direction=15
        else:
            raise ValueError(f"Unsupported camera pose conditioning type: {cond_type}")

        processed = rearrange(processed, "b t h w c -> b t c h w")
        processed = self._downsample_pose_time(processed) # align conditions with temporally downsampled latents
        return processed.to(dtype=input_dtype)
