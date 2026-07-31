from omegaconf import DictConfig

from algorithms.wan.modules.model import WanModel
from algorithms.wan.modules.model_pose import WanModelPose
from algorithms.wan.modules.model_wan22 import Wan22DiffSynthModel
from algorithms.wan.pose_utils import resolve_pose_conditioning_dim


def create_wan_model(backbone_cfg: DictConfig):
    model_family = str(getattr(backbone_cfg, "model_family", "wan21")).lower()
    if model_family in {"wan22", "wan2.2"}:
        return Wan22DiffSynthModel(
            has_image_input=bool(getattr(backbone_cfg, "has_image_input", False)),
            patch_size=tuple(backbone_cfg.patch_size),
            in_dim=int(backbone_cfg.in_channels),
            dim=int(backbone_cfg.dim),
            ffn_dim=int(backbone_cfg.ffn_dim),
            freq_dim=int(backbone_cfg.freq_dim),
            text_dim=int(backbone_cfg.text_dim),
            out_dim=int(backbone_cfg.out_channels),
            num_heads=int(backbone_cfg.num_heads),
            num_layers=int(backbone_cfg.num_layers),
            eps=float(backbone_cfg.eps),
            seperated_timestep=bool(getattr(backbone_cfg, "seperated_timestep", True)),
            require_clip_embedding=bool(
                getattr(backbone_cfg, "require_clip_embedding", False)
            ),
            require_vae_embedding=bool(
                getattr(backbone_cfg, "require_vae_embedding", False)
            ),
            fuse_vae_embedding_in_latents=bool(
                getattr(backbone_cfg, "fuse_vae_embedding_in_latents", False)
            ),
        )
    # Wan 2.1 1.3b pose or no pose
    else:
        pose_cfg = getattr(backbone_cfg, "pose_conditioning", None)
        pose_enabled = pose_cfg is not None and bool(getattr(pose_cfg, "enabled", False))
        model_cls = WanModelPose if pose_enabled else WanModel

        kwargs = dict(
            model_type=backbone_cfg.model_variant,
            patch_size=tuple(backbone_cfg.patch_size),
            text_len=int(backbone_cfg.text_len),
            in_dim=int(backbone_cfg.in_channels),
            dim=int(backbone_cfg.dim),
            ffn_dim=int(backbone_cfg.ffn_dim),
            freq_dim=int(backbone_cfg.freq_dim),
            text_dim=int(backbone_cfg.text_dim),
            out_dim=int(backbone_cfg.out_channels),
            num_heads=int(backbone_cfg.num_heads),
            num_layers=int(backbone_cfg.num_layers),
            window_size=tuple(backbone_cfg.window_size),
            qk_norm=bool(backbone_cfg.qk_norm),
            cross_attn_norm=bool(backbone_cfg.cross_attn_norm),
            eps=float(backbone_cfg.eps),
        )
        if pose_enabled:
            pose_type = str(getattr(pose_cfg, "type")).lower()
            kwargs.update(
                pose_conditioning_type=pose_type,
                pose_dim=resolve_pose_conditioning_dim(pose_type),
                pose_dropout_prob=float(getattr(pose_cfg, "dropout_prob", 0.0)),
            )
        return model_cls(**kwargs)
