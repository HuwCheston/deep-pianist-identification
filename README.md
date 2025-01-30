# Code from: Understanding Jazz Improvisation Style with Explainable Music Performer Identification Models 🤔💭🎹

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) ![coverage](coverage-badge.svg)
<a target="_blank" href="https://huwcheston.github.io/ImprovID-app/index.html">
<img src="https://img.shields.io/badge/Check%20out%20our%20webapp!-8A2BE2" alt="Check out our webapp!"/>
</a>

This repo accompanies our paper "Understanding Jazz Improvisation Style with Explainable Music Performer Identification
Models". For more information, see [our preprint](TODO) or check out
the [interactive web application.](https://huwcheston.github.io/ImprovID-app/index.html)

The code in this repository was developed using the following configuration:

- Ubuntu 22.04.1
- Python 3.10.12
- CUDA 12.2

## Contents:

- [Setup](#setup)
- [Reproducing results and figures](#reproducing-results-and-figures)
- [Tests](#tests)
- [License](#license)
- [Citation](#citation)

## Setup

First, clone the repository and install the dependencies in the usual way:

```
git clone https://github.com/HuwCheston/deep-pianist-identification.git
python -m venv venv    # use python3.10
source venv/bin/activate
pip install -r requirements.txt
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
    │           └── checkpoint_099.pth   # checkpoint of best factorised model
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

## Reproducing results and figures

- To reproduce the results from the handcrafted features models described in section 3. of our paper,
  see [this README](deep_pianist_identification/whitebox/README.md).
- To train and reproduce the results for the neural network architectures described in section 4. and 5. of our paper (
  including our factorized architecture), see [this README](deep_pianist_identification/encoders/README.md).
- To reproduce the explainability techniques applied in sections 4.2.2. 5.2.4.,
  see [this README](deep_pianist_identification/explainability/README.md)
- Finally, if (for whatever reason) you want to
  rebuild [our web application](https://huwcheston.github.io/ImprovID-app/index.html), you can check
  out [this README](deep_pianist_identification/app/README.md)

## Tests

To run all the tests, follow the steps above to [download the data](#setup). Then, you can run:

```
coverage run -m unittest discover && coverage xml -i coverage.xml && genbadge coverage -i coverage.xml
```

## License

This code is licensed under the [MIT license](LICENSE).

## Citation

If you refer to any aspect of this work, please cite the following preprint:

<details>
<summary>View citation</summary>

```
@article{cheston2025jazz,
  title = {Understanding Jazz Improvisation Style with Explainable Music Performer Identification Models},
  author = {Huw Cheston and Reuben Bance and Peter Harrison},
  journal = {arXiv},
  year = {2025},
  eprint = {arXiv:TODO},
  eprinttype = {arxiv},
  eprintclass = {cs.SD},
  institution = {Centre for Music and Science, University of Cambridge}
}
```

</details>
