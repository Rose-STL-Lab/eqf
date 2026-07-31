# Configuration

EqF uses [Hydra](https://hydra.cc/docs/intro/) to compose the dataset,
algorithm, experiment, checkpoint map, cluster settings, and a user-facing
shortcode. The root defaults are in `configurations/config.yaml`.

Release shortcodes are organized by dataset and method:

- `configurations/shortcode/exp/minecraft/{eqf,eqm,flowf,dfot}/`
- `configurations/shortcode/exp/re10k/{eqf,flowf}/`
- `configurations/shortcode/exp/droid/{eqf,flowf}/`
- `configurations/shortcode/preprocess/`

A shortcode selects its algorithm and dataset internally, which can also be overridden through the command line:

```bash
python -m main \
  shortcode=exp/minecraft/eqf/train_eqf \
  ckpt_map=default \
  +name=minecraft_eqf
```

Canonical public inference recipes, using the default checkpoint map, are:

- `shortcode=exp/minecraft/eqf/infer`
- `shortcode=exp/re10k/eqf/infer`
- `shortcode=exp/droid/eqf/infer_adaptive`


## Command line overrides

Append `key=value` pairs to change a composed value. For example:

```bash
python -m main \
  shortcode=exp/minecraft/eqf/train_eqf \
  ckpt_map=default \
  +name=minecraft_eqf \
  experiment.devices=4 \
  experiment.training.batch_size=4
```

Use slash syntax to select a config group. VAE configs are selected with
`algorithm/vae=<name>`:

```bash
python main.py \
  experiment=visualize_dataset_and_vae \
  dataset=minecraft \
  algorithm/vae=image_vae_minecraft \
  ckpt_map=default \
  +name=viz_minecraft_vae
```

`algorithm.vae=<name>` is not equivalent: it replaces the structured VAE node
with a string rather than composing the VAE config group.

## Checkpoint and cluster maps

`configurations/ckpt_map/default.yaml` defines the portable public checkpoint
layout. One may also define a custom save location for outputs, and custom checkpoint layouts using a new file here.

Every resolved Hydra configuration is saved with its W&B run. This makes the
shortcode plus command-line overrides the complete record of a launched job.

