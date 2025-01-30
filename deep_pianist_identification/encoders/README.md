# Training the neural networks

This README walks through the process of training the neural networks described in sections 4. and 5. of our paper.

## Config files

First, make sure you've followed the *setup* section on [this README](../../README.md)

We use `.yaml` files to store the configuration for every experiment. They are stored inside `./config`.

<details>
<summary>Here is an annotated example:</summary>

```
experiment: default    # propagated to MLFlow
run: debug_local    # propagated to MLFlow
batch_size: 4
epochs: 100
classify_dataset: false    # not currently used, should always be set to false
data_split_dir: 20class_80min    # use the data splits described in the paper
train_dataset_cfg:
  n_clips: 100    # sets a ceiling on number of clips: if null, will use all clips
  multichannel: false    # set to true to use factorized representations
  normalize_velocity: true    # normalize velocity between 0. and 1.
test_dataset_cfg:    # will be propagated to validation dataloader as well
  n_clips: 100
  multichannel: false
  normalize_velocity: true
encoder_module: cnn    # encoder module to use: can be {cnn, crnn, resnet50, disentangle}, where disentangle === factorised model
model_cfg: { }    # pass kwargs to encoder module here
optim_type: adam    # can be {adam, sgd}
optim_cfg:    # pass kwargs to optimizaer here
  lr: 1.0e-3
sched_type: null    # can be {plateau, cosine, step, linear, null}
loss_type: cce    # can be {cce, cce+triplet}, note that only cce was used in our paper
loss_cfg: { }    # arguments passed to loss function
checkpoint_cfg:
  save_checkpoints: false    # checkpoint during training
  load_checkpoints: false    # train from scratch
  checkpoint_after_n_epochs: 10    # save checkpoints after 10 epochs
  checkpoint_dir: ../checkpoints    # dump checkpoints to checkpoint_dir/experiment/run
mlflow_cfg:
  use: false
  tracking_uri: ...    # here is where the mlflow server is located, must be reachable
  run_id: ...    # pass this to resume a previous mlflow run, otherwise will start a new run
```

</details>

## CLI for training

We have a command line interface for training the models. Simply run:

```
python deep_pianist_identification/training.py --config <path-from-./config>
```

The `--config` parameter must be relative to the `./config` directory of the repository. So, to train a CRNN without
augmentation, I'd run:

```
python deep_pianist_identification/training.py --config baselines/crnn-jtd+pijama.yaml
```

The command line interface will log a bunch of metrics at the start of every training run (e.g., number of parameters)
which you can use to sanity check the arguments passed in your `.yaml` file. It will also report the number of batches
processed within every epoch and log the accuracy and loss at the end of each epoch. Finally, after every 5 epochs
you'll get a confusion matrix saved inside `checkpoint_dir/experiment/run/figures` to help you see the per-class
accuracy of the model.

## Experiments described in the paper

*Table 1*

These experiments are for the **factorized** model

| Encoder   | Heads | Pooling | Augment | Mask | Test accuracy | Filepath (relative to `./config`)                                                                                    |
|-----------|-------|---------|---------|------|---------------|----------------------------------------------------------------------------------------------------------------------|
| ResNet-18 | 0     | Average | Y       | Y    | *0.906*       | `disentangle-resnet-channel/disentangle-jtd+pijama-resnet18-mask30concept3-augment50-noattention-avgpool-onefc.yaml` |
| ResNet-34 | 0     | Average | Y       | Y    | *0.906*       | `disentangle-final-experiments/disentangle-jtd+pijama-resnet34-noattention-avgpool-augment50-mask30concept3.yaml`    |
| ResNet-50 | 0     | Average | Y       | Y    | 0.863         | `disentangle-final-experiments/disentangle-jtd+pijama-resnet50-noattention-avgpool-augment50-mask30concept3.yaml`    |
| CRNN      | 0     | Average | Y       | Y    | 0.812         | `disentangle-final-experiments/disentangle-jtd+pijama-8conv3pool-noattention-avgpool-augment50-mask30concept3.yaml`  |
| ResNet-18 | 4     | Average | Y       | Y    | 0.825         | `disentangle-final-experiments/disentangle-jtd+pijama-resnet18-4heads-avgpool-augment50-mask30concept3.yaml`         |
| ResNet-18 | 8     | Average | Y       | Y    | 0.856         | `disentangle-final-experiments/disentangle-jtd+pijama-resnet18-8heads-avgpool-augment50-mask30concept3.yaml`         |
| ResNet-18 | 16    | Average | Y       | Y    | 0.844         | `disentangle-final-experiments/disentangle-jtd+pijama-resnet18-16heads-avgpool-augment50-mask30concept3.yaml`        |
| ResNet-18 | 0     | Max     | Y       | Y    | 0.906         | `disentangle-final-experiments/disentangle-jtd+pijama-resnet18-noattention-maxpool-augment50-mask30concept3.yaml`    |
| ResNet-18 | 4     | Max     | Y       | Y    | 0.831         | `disentangle-final-experiments/disentangle-jtd+pijama-resnet18-4heads-maxpool-augment50-mask30concept3.yaml`         |
| ResNet-18 | 0     | Average | N       | Y    | 0.789         | `disentangle-final-experiments/disentangle-jtd+pijama-resnet18-noattention-avgpool-noaugment-mask30concept3.yaml`    |
| ResNet-18 | 0     | Average | Y       | N    | 0.863         | `disentangle-resnet-channel/disentangle-jtd+pijama-resnet18-nomask-augment50-noattention-avgpool-onefc.yaml`         |
| ResNet-18 | 0     | Average | N       | N    | 0.806         | `disentangle-final-experiments/disentangle-jtd+pijama-resnet18-noattention-avgpool-noaugment-nomask.yaml`            |

When training the factorised model, you'll probably find that -- depending on the settings passed in the `.yaml` file --
you'll end up with `ExtractorError`s for some clips. This is normal, and simply refers to cases where e.g., the harmony
extraction algorithm wasn't able to extract any chords from input MIDI. These clips will be skipped during training.

*Table S5*

These experiments are for the **non-factorized** model

| Encoder   | Augment | Test accuracy | Filepath (relative to `./config`)            |
|-----------|---------|---------------|----------------------------------------------|
| CRNN      | N       | 0.769         | `baselines/crnn-jtd+pijama.yaml`             |
| CRNN      | Y       | 0.825         | `baselines/crnn-jtd+pijama-augment.yaml`     |
| ResNet-50 | N       | 0.875         | `baselines/resnet50-jtd+pijama.yaml`         |
| ResNet-50 | Y       | 0.944         | `baselines/resnet50-jtd+pijama-augment.yaml` |

<details>
<summary>A note on `mlflow`</summary>

Training is set up to log metrics (loss, accuracy) on `mlflow`. However, I had to make a few hard-coded assumptions for
the systems I was using to train these models. If you want to use mlflow, you'll probably need to change the
`get_tracking_uri` function in `deep_pianist_identification/training.py` to point towards your `mlflow` server and port.

Otherwise, you can just set `mlflow_cfg["use"]: False` in the `.yaml` file to ignore `mlflow`: you'll still get metrics
logged to the command line during training.

</details>
