# Code from: Machine Learning of Artistic Fingerprints in Jazz 🤔💭🎹

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![coverage](coverage-badge.svg)
[![Check out our webapp!](https://img.shields.io/badge/Check%20out%20our%20webapp!-8A2BE2)](https://huwcheston.github.io/ImprovID-app/index.html)
[![Model card](https://img.shields.io/badge/Model%20card-8A2BE2)](https://github.com/HuwCheston/deep-pianist-identification/blob/main/modelcard.md)

This repository accompanies our paper "Machine Learning of Artistic Fingerprints in Jazz". For more information, see [our preprint](https://arxiv.org/abs/2504.05009) or check out
the [interactive web application.](https://huwcheston.github.io/ImprovID-app/index.html)

The code in this repository was developed and tested using the following configuration:

- Ubuntu 22.04.1
- Python 3.10.12
- CUDA 12.2
- Poetry

Full Python dependencies can be found inside the [`pyproject.toml` file](https://github.com/HuwCheston/deep-pianist-identification/blob/main/requirements.txt).

## Contents:

- [Setup](#setup)
- [Reproducing results and figures](#reproducing-results-and-figures)
- [Demo](#demo)
- [Tests](#tests)
- [License](#license)
- [Citation](#citation)

## Setup

First, clone the repository and install the dependencies in the usual way:

```
git clone https://github.com/HuwCheston/deep-pianist-identification.git
poetry install
```

Then, you can download the data, model checkpoints, and additional resources
from [our Zenodo archive](https://zenodo.org/records/14774191) as a `.zip`
file. The folder structure of the `.zip` is identical to this repository, so if you unzip it to the root directory (
`deep-pianist-identification`), you should end up with something like the following:

<details>
<summary>View filestructure</summary>

```
.
└── deep-pianist-identification/
    ├── data/
    │   ├── clips/                # pre-truncated 30 second clips (download from Zenodo)
    │   │   ├── pijama/
    │   │   │   ├── one_folder_per_track
    │   │   │   └── ...
    │   │   └── jtd/
    │   │       ├── one_folder_per_track
    │   │       └── ...
    │   └── raw/                  # metadata and full performances (download from Zenodo)
    │       ├── pijama
    │       └── jtd
    ├── checkpoints/
    │   ├── baselines/
    │   │   └── crnn-jtd+pijama-augment/
    │   │       └── checkpoint_099.pth    # checkpoint of best CRNN
    │   │   └── resnet50-jtd+pijama-augment/
    │   │       └── checkpoint_099.pth    # checkpoint of best resnet
    │   └── disentangle-resnet-channel/
    │       └── disentangle-jtd+pijama-resnet18-mask30concept3-augment50-noattention-avgpool-onefc/
    │           └── checkpoint_099.pth   # checkpoint of best multi-input model
    ├── references/
    │   ├── cav_resources/
    │   │   └── voicings/
    │   │       └── midi_final/
    │   │           ├── 1_cav/            # one folder per CAV
    │   │           │   ├── 1.mid
    │   │           │   └── 2.mid
    │   │           ├── 2_cav/
    │   │           │   └── ...
    │   │           └── ...                # Download these examples from Zenodo
    └── reports/
        └── figures/           # raw files for results in our paper
```

</details>

The time typically required to install the repository and all dependencies on a "normal" desktop computer is under 10 minutes.

## Reproducing results and figures

- To reproduce the results from the handcrafted features models described in section 3. of our paper,
  see [this README](deep_pianist_identification/whitebox/README.md).
- To train and reproduce the results for the neural network architectures described in section 4. and 5. of our paper (including our multi-input architecture), see [this README](deep_pianist_identification/encoders/README.md).
- To reproduce the explainability techniques applied in sections 4.2.2. 5.2.4.,
  see [this README](deep_pianist_identification/explainability/README.md)
- Finally, if (for whatever reason) you want to
  rebuild [our web application](https://huwcheston.github.io/ImprovID-app/index.html), you can check
  out [this README](deep_pianist_identification/app/README.md)

## Demo

To quickly run inference on the held-out test data using our pre-trained multi-input model, follow the instructions given in [Setup](#setup), then run the following command:

```
python deep_pianist_identification/validation.py --config disentangle-resnet-channel/disentangle-jtd+pijama-resnet18-mask30concept3-augment50-noattention-avgpool-onefc.yaml
```

Once finished, outputs will be saved to `reports/figures/ablated_representations`. Inference requires a GPU with at least 4 GB of VRAM. On a "normal" desktop computer with a NVIDIA RTX 3080 TI GPU, it takes roughly 90 seconds to run this script.

## Tests

To run all the tests, follow the steps above to [download the data](#setup). Then, you can run:

```
pip install coverage genbadge
coverage run -m unittest discover && coverage xml -i && genbadge coverage -i coverage.xml
```

## License

This code is licensed under the [MIT license](LICENSE).

## Citation

If you refer to any aspect of this work, please cite the following preprint:

<details>
<summary>View citation</summary>

```
@misc{cheston2025fingerprints,
      title={Machine Learning of Artistic Fingerprints in Jazz}, 
      author={Huw Cheston and Reuben Bance and Peter M. C. Harrison},
      year={2025},
      eprint={2504.05009},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2504.05009}, 
}
```

</details>
