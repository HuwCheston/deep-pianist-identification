# Running the explainability scripts

This README walks through the process of regenerating the LIME and TCAV analyses in sections 4.2.2. and 5.2.4 of our
paper.

## Running the scripts

Before running any of these scripts, make sure you've followed the *setup* section on [this README](../../README.md)

### LIME

To generate the LIME plots, you'll need to have trained the `baselines/resnet50-jtd+pijama-augment` model fully, or have
downloaded the checkpoints from Zenodo. Then, you just need to run:

```
python deep_pianist_identification/explainability/create_lime_plots.py
```

There are a few constants defined at the top-level of this file that you can adjust to change how these plots look. The
results will be dumped into `./results/figures/blackbox/lime_piano_rolls/<performer>`. You'll find one image generated
for every clip contained in the held-out test split of the dataset.

### TCAV

To generate the CAV outputs, you'll need to have trained the
`disentangle-resnet-channel/disentangle-jtd+pijama-resnet18-mask30concept3-augment50-noattention-avgpool-onefc.yaml`
model or download the checkpoints from Zenodo. You'll also need the CAV dataset from Zenodo. If you haven't done any of
this, make sure you've followed the *setup* section on [this README](../../README.md).

We have a command-line interface for generating CAVs + outputs which you can run with

```
python deep_pianist_identification/explainability/create_harmony_cavs.py
```

The CLI accepts the following arguments:

| Argument                      | Short Flag | Default Value             | Type         | Description                                                                    |
|-------------------------------|------------|---------------------------|--------------|--------------------------------------------------------------------------------|
| `--model`                     | `-m`       | see above                 | `str`        | Name of a trained model with saved checkpoints                                 |
| `--attribution-fn`            | `-a`       | `'gradient_x_activation'` | `str`        | Function to use to compute layer attributions (see captum.TCAV)                |
| `--multiply-by-inputs`        | `-i`       | `True`                    | `bool`       | Multiply layer activations by input (see captum.TCAV)                          |
| `--n-experiments`             | `-e`       | `10`                      | `int`        | Number of experiments to run (i.e., number of CAVs to create per concept)      |
| `--n-random-clips`            | `-r`       | `250`                     | `int`        | Number of clips to use when creating random CAV                                |
| `--n-cavs`                    | `-c`       | `20`                      | `int`        | Number of CAVs to create, defaults to 20                                       |
| `--batch-size`                | `-b`       | `10`                      | `int`        | Size of batches to use for GPU processing                                      |
| `--sensitivity-heatmap-clips` | `-s`       | "Tivoli" (see below)      | `str` (list) | Clips to create concept sensitivity heatmaps for, can be passed multiple times |

Generate CAVs will be stored as pickle files within `./reports/figures/ablated_representations/cav_plots`. On re-running
the CLI, *any saved Pickle files will be reloaded* to save on computation time. You'll need to delete or rename the
`*.p` files to avoid this.

After running the command, you'll find several `.json`, `.csv`, and `.png` files dumped inside
`./reports/figures/ablated_representations/cav_plots`. These are the figures and tables contained in our paper.

By default, the CLI will generate heatmaps for the clip at
`"pijama/tynerm-tivoli-unaccompanied-xxxx-zzs1xka8/clip_008.mid"`, which is shown in Figure 12 of our paper. You can
create additional heatmaps by passing further paths in to the `-s` flag.
