# Algorithms

The algorithm classes are Pytorch Lightning modules; shortcodes provide the dataset and method hyperparameters. 

## Equilibrium Forcing

EqF for minecraft is implemented in `algorithms/eqf_video.py`, for Wan 2.1 with Re10k it is `algorithms/wan_t2v_pose_eqf_video.py`, and for Wan 2.2 on Droid it is `algorithms/wan22_eqf_video.py`. Sampling supports fixed budgets, adaptive stopping from gradient norm or learned noise level readout, and budget adaptive inference. Sampling supports fixed budgets, adaptive stopping from
gradient norm or a learned noise-level readout, tail finish, and adaptive
rolling inference.

## Equilibrium Matching

EqM and EqF share training code, but EqM has the training equilibrium schedule set. Its canonical recipes with the truncated c function schedule and lambda=4 (as set in the EqM paper) are:

- `shortcode=exp/minecraft/eqm/train_trunc_lambda4`
- `shortcode=exp/minecraft/eqm/infer_trunc_lambda4`

## Flow matching

`algorithms/flowf_video.py` implements Minecraft Flow Matching.
`algorithms/wan_t2v_pose_forcing_video.py` implements the pose-conditioned Wan
2.1 model for RE10K, and `algorithms/wan22_forcing_video.py` implements Wan 2.2
flow matching for DROID.

## DFoT diffusion

`algorithms/dfot_video.py` is the Minecraft diffusion baseline.
Minecraft uses:

- `shortcode=exp/minecraft/dfot/train`
- `shortcode=exp/minecraft/dfot/infer`

## Noise-level-prediction EqF

NLP-EqF adds a learned noise-level readout to a trained EqF denoiser:

- `algorithms/noiselevelpred_eqf_video.py` for Minecraft.
- `algorithms/wan_pose_noiselevelpred_eqf_video.py` for RE10K.
- `algorithms/wan22_noiselevelpred_eqf_video.py` for DROID.

The finetuning recipes freeze the denoiser, and reset the optimizer state. The readout enables adaptive stopping based on predicted noise level.

## VAEs
Minecraft training is faster with precomputed ImageVAE latents. Re10k and Droid are difficult to preprocess due to their causal asymmetric video vae format. 

- Minecraft: the released ImageVAE, selected with
  `algorithm/vae=image_vae_minecraft`.
- RE10K: Wan 2.1 VAE, selected with `algorithm/vae=wan`.
- DROID: Wan 2.2 VAE, selected with `algorithm/vae=wan22`.

