# Training

Prepare the relevant data before training, see [[Dataset]]. Override `+name=` to set the name, and enable wandb logging by setting `wandb.mode=online`.

## Minecraft

Minecraft has four base training recipes:

```bash
# Equilibrium Forcing
python -m main shortcode=exp/minecraft/eqf/train_eqf ckpt_map=default +name=minecraft_eqf

# Flow matching
python -m main shortcode=exp/minecraft/flowf/train ckpt_map=default +name=minecraft_fm

# Equilibrium Matching (truncated schedule as per the EqM paper, lambda=4)
python -m main shortcode=exp/minecraft/eqm/train_trunc_lambda4 ckpt_map=default +name=minecraft_eqm

# DFoT diffusion
python -m main shortcode=exp/minecraft/dfot/train ckpt_map=default +name=minecraft_dfot
```

These recipes encode Minecraft videos with the ImageVAE online by default. If
you have run the optional preprocessing command from [[Dataset]], append
`dataset.latent.type=pre_sample` to use the cached latents and improve training
throughput. In our experience this can speed up training up to 2x.

After EqF training completes, use this command to train the Minecraft noise level prediction readout, overriding the `load=...` based on where your checkpoint is saved.
```bash
python -m main shortcode=exp/minecraft/eqf/train_noiselevel \
  ckpt_map=default \
  +name=minecraft_nlp_eqf \
  load=/path/to/minecraft_eqf.ckpt
```

## RE10K

RE10K uses pose-conditioned Wan 2.1 models:

```bash
# Equilibrium Forcing
python -m main shortcode=exp/re10k/eqf/train_pose ckpt_map=default +name=re10k_eqf

# Flow matching
python -m main shortcode=exp/re10k/flowf/train_pose ckpt_map=default +name=re10k_fm
```

Finetune the noise level prediction readout from the base pose-conditioned EqF checkpoint. Override `load=...` based on where your checkpoint is saved:

```bash
python -m main shortcode=exp/re10k/eqf/train_pose_noiselevel \
  ckpt_map=default \
  +name=re10k_nlp_eqf \
  load=/path/to/re10k_eqf.ckpt
```

## DROID

DROID uses Wan 2.2 and cached prompt embeddings. Make sure you have run the caching step from the repo's README first.:

```bash
# Equilibrium Forcing
python -m main shortcode=exp/droid/eqf/train ckpt_map=default +name=droid_eqf

# Flow matching
python -m main shortcode=exp/droid/flowf/train ckpt_map=default +name=droid_fm
```

Then train the noise level prediction readout model from the base EqF checkpoint (the shortcode obtains the base EqF checkpoint from `${algorithm.model_weights.droid.eqf}`; override based on your trained model in the shortcode):

```bash
python -m main shortcode=exp/droid/eqf/train_noiselevel ckpt_map=default +name=droid_nlp_eqf
```


## NLP-EqF finetuning behavior

The NLP recipes switch to the dataset-specific noise-level-prediction algorithm,
freeze the denoising model, allow the new readout keys to be absent from the
base checkpoint, and reset optimizer state. DROID explicitly bootstraps the
frozen denoiser from the base checkpoint's EMA weights. Minecraft and RE10K
accept the base checkpoint through `load=`.

Use `resume=<wandb-run-id>` only when reconnecting logging to an existing W&B
run. Use `load=<checkpoint>` for the Lightning model/trainer state described
above.
