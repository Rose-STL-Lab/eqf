from typing import Optional

import torch
from torch import Tensor


class DataMixin:
    """Hooks for shaping the conditioning tensor as it flows through the pipeline.

    Two hooks, separated by *when* they fire:

    - `_encode_conditions`: called once in `on_after_batch_transfer` to turn the
      raw dataset feature tensor into the model-ready representation. Used by
      e.g. the pose mixin to convert raw `(B, T, 16)` cameras into ray maps on
      the latent grid. Default: identity.

    - `_mask_window_conditions`: called per streaming window after slicing and
      padding, right before handing the slice to the denoising algorithm. Used
      for context-boundary masking (`mask_first` / `mask_last`) and could be
      extended to per-window CFG-style dropout. Default: applies the legacy
      `external_cond_processing` setting if present, otherwise identity.

    `_slice_and_pad_window_conditions` is a small pure helper used by streaming
    inference to crop a window out of the global conditions tensor and zero-pad
    it to the model's forward window size.
    """

    def _encode_conditions(self, conditions: Optional[Tensor]) -> Optional[Tensor]:
        """One-time conversion from raw dataset conditions to model-ready form.

        Subclasses override this to perform expensive clip-global preprocessing
        such as building ray maps from raw camera vectors. Runs in
        `on_after_batch_transfer`. Default: identity.
        """
        return conditions

    @torch.no_grad()
    def _mask_window_conditions(
        self, conditions: Optional[Tensor]
    ) -> Optional[Tensor]:
        """Per-window dropout / boundary masking applied to a sliced window.

        Runs once per streaming window, after slicing and padding. Default
        applies the `external_cond_processing` config (`mask_first` /
        `mask_last`) if set; otherwise it is a no-op.
        """
        if conditions is None or conditions.shape[-1] == 0:
            return conditions

        external_cond_processing = getattr(self.cfg, "external_cond_processing", None)
        if external_cond_processing is None:
            return conditions

        match external_cond_processing:
            case "mask_first":
                # First condition is meaningless under the
                # "fr_{i} + condition x_{i+1} -> fr_{i+1}" formulation: there is
                # no fr_{-1}, so we zero out the first frame's condition.
                mask = torch.ones_like(conditions)
                mask[:, :1, : self.external_cond_dim] = 0
                return conditions * mask
            case "mask_last":
                mask = torch.ones_like(conditions)
                mask[:, -1, : self.external_cond_dim] = 0
                return conditions * mask
            case _:
                raise NotImplementedError(
                    f"External condition processing {external_cond_processing} is not implemented."
                )

    @staticmethod
    def _slice_and_pad_window_conditions(
        conditions: Optional[Tensor],
        start: int,
        end: int,
        target_len: int,
    ) -> Optional[Tensor]:
        """Slice the time axis to `[start, end)` and zero-pad to `target_len`."""
        if conditions is None:
            return None
        sliced = conditions[:, start:end]
        if sliced.shape[1] >= target_len:
            return sliced
        pad = sliced.new_zeros(
            sliced.shape[0],
            target_len - sliced.shape[1],
            *sliced.shape[2:],
        )
        return torch.cat([sliced, pad], dim=1)
