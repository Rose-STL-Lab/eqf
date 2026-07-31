# Datasets

We release code for handling the Minecraft, Re10k, and Droid datasets. 
Dataset roots default to `data/<dataset>` and can be changed with
`dataset.save_dir=/path/to/data`. See the main repo's README for dataset download and preprocessing commands.

## Minecraft

The Minecraft dataset is action conditioned and uses a framewise pretrained image VAE. 
Training encodes videos online by default, so only the prepared videos under
`data/minecraft` and the ImageVAE path in `ckpt_map/default.yaml` are required. 

Download with 

```bash
bash scripts/download_minecraft.sh --workers 11
```

To improve training throughput, optionally precompute latents with:

```bash
python -m main \
  shortcode=preprocess/preprocess_minecraft \
  experiment=video_latent_preprocessing \
  algorithm=image_vae_preprocessor \
  dataset=minecraft \
  ckpt_map=default \
  +name=preprocess_minecraft
```

After preprocessing, add `dataset.latent.type=pre_sample` to a Minecraft
training command or change the shortcode directly. Without that override, the same shortcodes use `dataset.latent.type=online` and encode each batch with the VAE during training.

## RE10K

Re10k uses pose conditioning at 10 FPS, and uses online Wan 2.1 VAE encoding. Its default root is `data/re10k`. Use the scripts
`scripts/download_re10k.sh` and `scripts/make_metadata_re10k.py` for preparing
videos and metadata:

```bash
bash scripts/download_re10k.sh
python scripts/make_metadata_re10k.py
```

## DROID

DROID uses online Wan 2.2 VAE encoding. The dataset root defaults to
`data/droid`, with the source dataset under `data/droid/1.0.1`. Training expects
`cleaned_metadata_with_prompt_embeds.csv`, which is also downloaded through the script, originating from the LVP paper:

```bash
bash scripts/download_droid.sh
```

Starting from `cleaned_metadata.csv`, cache prompt embeddings with:

```bash
python -m main \
  experiment=cache_prompt_embeds \
  dataset=droid \
  algorithm=wan22_forcing_video \
  ckpt_map=default \
  dataset.metadata_path=cleaned_metadata.csv \
  experiment.cache_prompt_embeds.batch_size=256 \
  +name=cache_droid_prompt_embeds
```

## Check dataset and VAE decoding

Use these commands to visualize a batch and its VAE reconstruction:

```bash
python main.py experiment=visualize_dataset_and_vae dataset=minecraft \
  ckpt_map=default algorithm/vae=image_vae_minecraft \
  +name=viz_minecraft_vae
python main.py experiment=visualize_dataset_and_vae dataset=re10k \
  ckpt_map=default algorithm/vae=wan \
  +name=viz_re10k_vae
python main.py experiment=visualize_dataset_and_vae dataset=droid \
  ckpt_map=default algorithm/vae=wan22 \
  experiment.visualize.bf16=true \
  +name=viz_droid_vae
```
