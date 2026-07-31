from .cogv_transformers import CogVideoXTransformer3DModel
from typing import Optional, List, Literal

__all__ = ["create_cogv_like_model"]

def create_cogv_like_model(
        num_attention_heads: int = 30,
        attention_head_dim: int = 64,
        in_channels: int = 16,
        out_channels: Optional[int] = 16,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        time_embed_dim: int = 512,
        external_cond_dim: int = 4096,
        num_layers: int = 30,
        dropout: float = 0.0,
        attention_bias: bool = True,
        sample_width: int = 32,
        sample_height: int = 18,
        forward_window_size: int = 13,
        patch_size: int = 2,
        temporal_compression_ratio: int = 1,
        activation_fn: str = "gelu-approximate",
        timestep_activation_fn: str = "silu",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        spatial_interpolation_scale: float = 1.875,
        temporal_interpolation_scale: float = 1.0,
        use_rotary_positional_embeddings: bool = False,
        use_learned_positional_embeddings: bool = False,
        control_dim: int = 768,
        control_start_layer: int = 0,
        control_end_layer: int = 30,
        control_zero_init: bool = True,
        trainable_params: Optional[List[str]] = None,
        model_type: Literal["cogv"] = "cogv",
        use_depth: bool = False,
        # User-facing knob: the *VAE* may be non-causal, but we may want to treat its
        # temporal compression as "causal-like" for token accounting / positional embeddings.
        does_vae_act_causal: bool = False,
        **kwargs,
    ):
    AVAILABLE_MODELS = ["cogv"]
    if model_type == "cogv":
        model_class = CogVideoXTransformer3DModel
    else:
        raise ValueError(f"Invalid model type: {model_type}. Available models: {AVAILABLE_MODELS}")

    return model_class(
        num_attention_heads=num_attention_heads,
        attention_head_dim=attention_head_dim,
        in_channels=in_channels,
        out_channels=out_channels,
        flip_sin_to_cos=flip_sin_to_cos,
        freq_shift=freq_shift,
        time_embed_dim=time_embed_dim,
        external_cond_dim=external_cond_dim,
        num_layers=num_layers,
        dropout=dropout,
        attention_bias=attention_bias,
        sample_width=sample_width,
        sample_height=sample_height,
        forward_window_size=forward_window_size,
        patch_size=patch_size,
        temporal_compression_ratio=temporal_compression_ratio,
        does_vae_act_causal=does_vae_act_causal,
        activation_fn=activation_fn,
        timestep_activation_fn=timestep_activation_fn,
        norm_elementwise_affine=norm_elementwise_affine,
        norm_eps=norm_eps,
        trainable_params=trainable_params,
        use_rotary_positional_embeddings=use_rotary_positional_embeddings,
        use_depth=use_depth,
        **kwargs,
    )
