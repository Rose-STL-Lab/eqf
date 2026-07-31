# Equilibrium Forcing

This wiki documents the training and inference code for Equilibrium Forcing
(EqF) on Minecraft, RealEstate10K (RE10K), and DROID.

The release includes:

- EqF, flow matching (FM), EqM, and DFoT diffusion on Minecraft.
- Pose-conditioned EqF and FM finetuning using Wan 2.1 on RE10K.
- EqF and FM finetuning using Wan 2.2 on DROID.
- Noise-level-prediction EqF (NLP-EqF) readout finetuning for all three
  datasets.
- Fixed budget, adaptive early stopping, and budget adaptive inference.

The codebase uses [Hydra](https://hydra.cc/) for hierarchical configuration and
[PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) for training and
evaluation. Its four main configuration components are:

- `dataset`: data loading, preprocessing, and conditioning settings.
- `algorithm`: the model, denoising objective, training loss, and sampling.
- `experiment`: trainer, hardware specifications, etc.
- `shortcode`: dataset/method recipes.

It's recommended to start from a shortcode, and then override anything necessary from the command line.
```bash
python -m main shortcode=exp/minecraft/eqf/train_eqf ckpt_map=default +name=minecraft_eqf
```

## Documentation

- [[Training]]: Base model and noise level prediction finetuning commands.
- [[Dataset]]: Information about datasets, preprocessing, local paths, etc.
- [[Configurations]]: Hydra configurations and command line overrides.
- [[Algorithms]]: EqF, baselines, Wan variants, VAEs, etc.
- [[Inference and Reproducing Results]]: Full inference and table replication commands.

This repository is based on [FloWM](https://github.com/hlillemark/flowm) and also on Boyuan Chen's
[research template](https://github.com/buoyancy99/research-template).
