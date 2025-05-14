# Model Card for "multi-input jazz performer identification model"

This model is capable of identifying twenty jazz pianists from symbolic (MIDI) representations of their performances, and explaining its predictions across four fundamental musical domains -- harmony, melody, rhythm, and dynamics.

## Model Details

### Model Description

- **Developed by:** researchers at the [Centre for Music and Science, University of Cambridge](https://cms.mus.cam.ac.uk/)
- **Funded by:** [Cambridge Trust](https://www.cambridgetrust.org/)
- **Shared by:** [Huw Cheston](https://huwcheston.github.io/)
- **Model type:** convolutional neural network
- **License:** [MIT](https://github.com/HuwCheston/deep-pianist-identification?tab=MIT-1-ov-file#readme)

### Model Sources

- **Repository:** https://github.com/HuwCheston/deep-pianist-identification
- **Paper:** https://arxiv.org/abs/2504.05009
- **Demo:** https://cms.mus.cam.ac.uk/jazz-piano-style-ml

## Uses

### Direct Use

- Identification of twenty famous jazz pianists from symbolic (MIDI) representations of their performances
- Explainable modelling of jazz improvisation style across four dimensions (harmony, melody, rhythm, dynamics)
- Probing sensitivity of individual MIDI recordings to particular harmonic concepts, e.g. `Modal Voicings`, `II-V-I Progressions`.

### Downstream Use

- Retrieval of similar MIDI files across specific musical dimensions (harmony, melody, etc.)
- Identification of non-jazz musicians, e.g. Western classical composers
- Identification of distinct jazz subgenres, e.g. hard-bop, cool jazz, jazz fusion (etc.)

### Out-of-Scope Use

- Music generation
- Application to non-symbolic musical representations, e.g. audio

## Bias, Risks, and Limitations

- While the transcriptions used to train this model are not themselves copyrighted, the audio recordings they are created from are likely under copyright.
- The training data is skewed heavily towards male jazz musicians active in the middle of the twentieth century, and may not actively represent the complete diversity of professional jazz pianists.

### Recommendations

- Further work needed to evaluate across a diverse range of musical styles, beyond jazz.
- An ideal evaluation dataset would additionally include annotations for the harmonic concepts we test as part of our TCAV approach.

## How to Get Started with the Model

To train the model from scratch:

- Follow the [installation instructions](https://github.com/HuwCheston/deep-pianist-identification/blob/main/README.md#setup)
- Download the training data from our [Zenodo archive](https://zenodo.org/records/14774191) as a `.zip` file and place inside the root directory of the repository.
- Run: `python deep_pianist_identification/training.py --config disentangle-resnet-channel/disentangle-jtd+pijama-resnet18-mask30concept3-augment50-noattention-avgpool-onefc.yaml`
- Evaluation metrics will be logged to the console during training.
- Once finished, a checkpoint will be saved inside `checkpoints/disentangle-resnet-channel/disentangle-jtd+pijama-resnet18-mask30concept3-augment50-noattention-avgpool-onefc`
- Further details can be found in [this README](https://github.com/HuwCheston/deep-pianist-identification/blob/main/deep_pianist_identification/encoders/README.md)

To run inference on the test data:

- Ensure the model has been trained, either by running the above code or by downloading the checkpoint from our [Zenodo archive](https://zenodo.org/records/14774191)
- Run: `python deep_pianist_identification/validation.py --config disentangle-resnet-channel/disentangle-jtd+pijama-resnet18-mask30concept3-augment50-noattention-avgpool-onefc.yaml`
- Once finished, outputs will be saved to `reports/figures/ablated_representations`

To reproduce the explainability techniques (TCAV):

- Ensure the model has been trained, either by running the above code or by downloading the checkpoint from our [Zenodo archive](https://zenodo.org/records/14774191)
- Run: `python deep_pianist_identification/explainability/create_harmony_cavs.py`
- Once finished, outputs will be saved to `reports/figures/ablated_representations/cav_plots`
- Further details can be found in [this README](https://github.com/HuwCheston/deep-pianist-identification/blob/main/deep_pianist_identification/explainability/README.md)

## Training Details

### Training Data

Trained on a subset of both the [Jazz Trio Database](https://github.com/HuwCheston/Jazz-Trio-Database) and [Piano Jazz with Automatic MIDI Annotations](https://github.com/almostimplemented/PiJAMA). Training data splits can be found [inside the repository](https://github.com/HuwCheston/deep-pianist-identification/tree/main/references/data_splits/20class_80min/train_split.csv). 

### Training Procedure

#### Preprocessing

- Recordings are split into 30-second "clips" with a hop of between 15 and 30 seconds (randomly assigned during training).
- Onset and offset times are "snapped" to the nearest 100 milliseconds.
- Velocity values are scaled linearly between 0 and 1, where 1 is the maximum velocity within the clip.
- Clips are converted to four separate "piano roll" representations using the `Pretty-MIDI` package.
  - These piano rolls each isolate either the harmony, melody, rhythm, and dynamics content of the input.

#### Training Hyperparameters

- **Training regime:** `fp32`
- **Learning rate:** `1e-4`
- **Batch size**: 20
- **Epochs**: 100

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Evaluated on a subset of both the [Jazz Trio Database](https://github.com/HuwCheston/Jazz-Trio-Database) and [Piano Jazz with Automatic MIDI Annotations](https://github.com/almostimplemented/PiJAMA). Testing and validation data splits can be found [inside the repository](https://github.com/HuwCheston/deep-pianist-identification/tree/main/references/data_splits/20class_80min/test_split.csv).

#### Factors

- Training and test data is stratified by performance context, with an equal number of solo piano and piano trio recordings included in both splits
- Evaluation is also disaggregated across the twenty performers and four musical concepts considered by the model (i.e., how accurate are predictions of Brad Mehldau when only considering harmony?)

#### Metrics

- Categorical cross-entropy loss
- Classification accuracy, for individual 30-second clips
- Classification accuracy, for complete recordings (**reported below**)

### Results

| Model | Multi-Input? | Test accuracy (recording) |
| ----- | ------------ | ------------------------- |
| Ours  | Yes          | 0.906                     |
| ResNet-50 (Kim et al., 2020) | No  | **0.944**                  |
| Logistic Regression | No  | 0.767                  |

#### Summary

Our model achieves significantly better accuracy in classifying performers than a non-neural approach. It performs slightly worse than a ResNet-50, which can be considered the current state-of-the-art for this task (Zhang et al, 2023), however it has the benefit of the multi-input architecture, which allows for a greater degree of interpretability.

## Model Examination

For extensive information on model interpretability, refer to section 5.2. [of our paper](https://arxiv.org/abs/2504.05009) and the [accompanying web application](https://cms.mus.cam.ac.uk/jazz-piano-style-ml).

## Environmental Impact

- **Hardware Type:** NVIDIA A100 SXM4 80 GB
- **Hours used:** 12.4
- **Cloud Provider:** [Cambridge High Performance Computing Wilkes3 Cluster](https://www.hpc.cam.ac.uk/high-performance-computing)
- **Compute Region:** `europe-west2` equivalent
- **Carbon Emitted:** 2.14 kg

## Citation

**BibTeX:**

```
@misc{cheston2025deconstructingjazzpianostyle,
      title={Deconstructing Jazz Piano Style Using Machine Learning}, 
      author={Huw Cheston and Reuben Bance and Peter M. C. Harrison},
      year={2025},
      eprint={2504.05009},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2504.05009}, 
}
```

**APA:**

```
Cheston, H., Bance, R., & Harrison, P. M. C. (2025). Deconstructing Jazz Piano Style Using Machine Learning (arXiv:2504.05009). arXiv. https://doi.org/10.48550/arXiv.2504.05009
```

## Model Card Authors

Huw Cheston

## Model Card Contact

`hwc31 AT cam.ac.uk`
