<h1 align="center">Equilibrium Forcing (EqF)</h1>

<p align="center">
  <p align="center">
    <a href="https://hansenlillemark.com/">Hansen Jin Lillemark*<sup>1</sup></a>
    ·
    <a href="https://a5rojas.github.io/">Alex Rojas*<sup>1</sup></a>
    ·
    <a href="https://zacharynovack.github.io/">Zachary Novack<sup>1</sup></a>
    ·
    <a href="https://raywang4.github.io/">Runqian Wang<sup>2</sup></a>
    <br/>
    <a href="https://yilundu.github.io/">Yilun Du<sup>3</sup></a>
    ·
    <a href="https://sites.google.com/view/yianma/home">Yian Ma<sup>1</sup></a>
    ·
    <a href="https://cseweb.ucsd.edu/~tberg/">Taylor Berg-Kirkpatrick<sup>1</sup></a>
    ·
    <a href="https://roseyu.com/">Rose Yu<sup>1</sup></a>
    <br/>
    *Equal contribution <sup>1</sup>UC San Diego <sup>2</sup>MIT <sup>3</sup>Harvard University
  </p>
  <h3 align="center"><a href="#">Paper</a> | <a href="https://equilibriumforcing.github.io/">Website</a> | <a href="https://huggingface.co/hlillemark/eqf">Models</a></h3>
</p>

Welcome to the codebase for the paper [Equilibrium Forcing: Adaptive Video Generation Without Noise Conditioning](https://equilibriumforcing.github.io)! EqF proposes to train video denoising generative models without noise level conditions. With an equilibrium field, EqF can leverage adaptive sampling algorithms to improve video generation performance, rather than following hand-designed rigid sampling schedules typically for video generation. The paper's analysis elucidates how noise-unconditional models can perform better than noise-conditional models (e.g. Flow Matching) through avoiding incorrect noise level conditions, how the two classes of models in fact learn the same underlying denoising velocity field, and how the EqF objective incentivizes the model to estimate the noise level of an input internally. 

This repository contains the training and inference code for EqF on Minecraft, RealEstate10K (Re10K), and Droid. It also contains the baseline code for standard Diffusion, Flow Matching, and Equilibrium Matching baselines.

Contents:
- [Environment Setup](#environment)
- [Quick Start](#quick-start)
- [Dataset Download](#dataset)
- [Model Checkpoint Download](#model-checkpoints)
- [Inference commands](#inference-commands)
- [Training commands](#training-commands)
- [Code Walkthrough](#code-walkthrough)
- [Citation](#citation)


# Setup

## Environment

`setup_env.sh` creates a conda environment named `eqf`. The script is meant for Cuda 13:

```bash
bash setup_env.sh
# or: bash setup_env.sh --cuda12.4
conda activate eqf
```

To track runs with wandb, make a copy of
`configurations/secrets/my_secrets_template.yaml` to
`configurations/secrets/my_secrets.yaml` and add your API key. Set the wandb entity and project in `configurations/config.yaml` or override them on the command line.


## Quick Start

To get started with inference, download a 256 video subset of the Minecraft validation data,
the Minecraft ImageVAE, and the EqF checkpoint:

```bash
bash scripts/download_minecraft_mini.sh
```

Generate 32 videos of 300 frames each with:

```bash
python -m main shortcode=exp/minecraft/eqf/infer \
  +name=minecraft_mini_quickstart dataset.save_dir=data/minecraft/mini \
  dataset.num_validation_clips=32
```

## Data

Data is expected under `data/<dataset>`. Download the full datasets with the following commands:

### Minecraft

This command downloads the minecraft dataset to `data/minecraft/{training,validation}`. It may take up to a full day to complete the download:
```bash
bash scripts/download_minecraft.sh --workers 11
```

Minecraft training encodes videos online with the released image VAE by default. For faster training, precompute the latents, which can be done in a command described [here](https://github.com/Rose-STL-Lab/eqf/wiki/Dataset).

### RE10K

Download and prepare RE10K to `data/re10k/{training,validation}` with:

```bash
bash scripts/download_re10k.sh
python scripts/make_metadata_re10k.py
```

### DROID

Download the raw DROID dataset and caption metadata with the first command. The
second command caches the captions with Wan2.2's UMT5 text-prompt embedding
model using all visible GPUs:

```bash
bash scripts/download_droid.sh

python -m main experiment=cache_prompt_embeds dataset=droid algorithm=wan22_forcing_video +name=cache_droid_prompt_embeds
```


## Model Checkpoints

Download all released checkpoints and dependencies (like pretrained VAEs). Instead of `all`, you can also pass in `{minecraft, re10k, droid}` to download just a subset of checkpoints:

```bash
bash scripts/download_checkpoints.sh all
```

The downloaded checkpoint layout is configured in
`configurations/ckpt_map/default.yaml`, which Hydra selects by default. EqF
checkpoints work for both closed (noise level prediction) and open loop
inference.

## Training

Experiments are run using hydra and pytorch lightning, specifying a dataset, algorithm, model family, and hyperparameters. By default, the commands will run on all available GPUs, and are configured for H100 training; adjust device counts and batch sizes for your hardware. Baseline commands are available in the [training guide](https://github.com/Rose-STL-Lab/eqf/wiki/Training).

```bash
# Minecraft EqF model
python -m main shortcode=exp/minecraft/eqf/train_eqf +name=minecraft_eqf

# RE10K EqF model
python -m main shortcode=exp/re10k/eqf/train_pose +name=re10k_eqf

# DROID EqF model
python -m main shortcode=exp/droid/eqf/train +name=droid_eqf
```

To then train a noise level prediction readout on top of a trained EqF model, there is an additional training command, requiring setting the trained model properly in the checkpoint map. The optimizer state is reset and is trained from the ema weights of the denoising model:

```bash
python -m main shortcode=exp/minecraft/eqf/train_noiselevel +name=minecraft_nlp_eqf load=/path/to/minecraft_eqf.ckpt
```


## Inference

Each canonical command evaluates 256 validation videos:

```bash
# Minecraft: EqF with NAG, 250 steps, 300 frame videos
python -m main shortcode=exp/minecraft/eqf/infer_nag +name=minecraft_nlp_eqf_nag_infer

# RE10K: EqF with NAG, 50 steps, 37 context + 152 generated frames
python -m main shortcode=exp/re10k/eqf/infer_nag +name=re10k_nlp_eqf_nag_infer

# DROID: budget-adaptive EqF with NAG, 50 steps, 13 context + 36 generated frames
python -m main shortcode=exp/droid/eqf/infer_adaptive +name=droid_eqf_budget_adaptive
```

These scripts launch a script to replicate the results in the primary tables with one seed and 256 videos. Use `--list` to inspect the method indices first, and then `--run-indices` to select a subset. See the [inference guide](https://github.com/Rose-STL-Lab/eqf/wiki/Inference-and-Reproducing-Results) for more details:

```bash
# Fixed ablations table on minecraft (250 steps) and re10k (50 steps)
bash scripts/minecraft_inference.sh
bash scripts/re10k_inference.sh

# Budget-adaptive comparison launchers for EqF NAG and FM at 10-50 steps
bash scripts/adaptive_minecraft_inference.sh
bash scripts/adaptive_re10k_inference.sh
bash scripts/adaptive_droid_inference.sh
```

## VAE dataset visualization

Visualize the dataset and VAE reconstruction using the following commands:
```bash
# Minecraft
python main.py experiment=visualize_dataset_and_vae dataset=minecraft \
  algorithm/vae=image_vae_minecraft \
  +name=viz_minecraft_vae
# Re10k
python main.py experiment=visualize_dataset_and_vae dataset=re10k \
  algorithm/vae=wan \
  +name=viz_re10k_vae
# Droid
python main.py experiment=visualize_dataset_and_vae dataset=droid \
  algorithm/vae=wan22 \
  experiment.visualize.bf16=true \
  +name=viz_droid_vae
```


## Code Walkthrough

Hydra configurations live under `configurations/`. User-facing recipes are
under `configurations/shortcode/`; algorithm, dataset, experiment, checkpoint,
and cluster configuration groups can also be selected directly. See
`configurations/README.md` and the
[configuration guide](https://github.com/Rose-STL-Lab/eqf/wiki/Configurations)
for conventions.

The [full code wiki](https://github.com/Rose-STL-Lab/eqf/wiki) contains detailed information about the code structure and full training and inference commands.

## Citation

```bibtex
todo
```

## Acknowledgement

This repo is forked from [FloWM](https://github.com/hlillemark/flowm), which is based on Boyuan Chen's research template [repo](https://github.com/buoyancy99/research-template)
