# Code from: Understanding Jazz Improvisation Style with Explainable Music Performer Identification Models 🎹🎻🥁

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) <a target="_blank" href="https://huwcheston.github.io/ImprovID-app/index.html">
<img src="https://img.shields.io/badge/Check%20out%20our%20webapp!-8A2BE2" alt="Check out our webapp!"/>
</a>

This repo accompanies our paper "Understanding Jazz Improvisation Style with Explainable Music Performer Identification
Models". For more information, see [our preprint](TODO) or check out
the [interactive web application.](https://huwcheston.github.io/ImprovID-app/index.html)

## Contents:

- [Setup](#setup)
- [License](#license)
- [Citation](#citation)

## Setup

First, clone the repository and install the dependencies in the usual way:

```
git clone https://github.com/HuwCheston/deep-pianist-identification.git
python -m venv venv    # not necessary but advised
source venv/bin/activate
pip install -r requirements.txt
```

Then, you can download the data, model checkpoints, and additional resources from [our Zenodo archive](TODO). Extract
the files into the following directories

```
.
└── deep-pianist-identification/
    ├── data/
    │   ├── clips/    # pre-truncated 30 second clips (download from Zenodo)/
    │   │   ├── pijama/
    │   │   │   ├── one_folder_per_track
    │   │   │   └── ...
    │   │   └── jtd/
    │   │       ├── one_folder_per_track
    │   │       └── ...
    │   └── raw/    # metadata and full performances (download from Zenodo)/
    │       ├── pijama
    │       └── jtd
    ├── checkpoints/
    │   ├── baselines/
    │   │   └── resnet50-jtd+pijama-augment/
    │   │       └── checkpoint_099.pth    # checkpoint of best non-factorised model, download from Zenodo
    │   └── disentangle-resnet-channel/
    │       └── disentangle-jtd+pijama-resnet18-mask30concept3-augment50-noattention-avgpool-onefc/
    │           └── checkpoint_099.pth   # checkpoint of best factorised model, download from Zenodo
    └── references/
        └── cav_resources/
            └── voicings/
                └── midi_final/
                    ├── 1_cav    # one folder per CAV/
                    │   ├── 1.mid
                    │   └── 2.mid
                    ├── 2_cav/
                    │   └── ...
                    └── ... # Download these examples from Zenodo
```

## Reproducing figures

TODO

## License

This code is licensed under the [MIT license](LICENSE.md).

## Citation

If you refer to any aspect of this work, please cite the following preprint:

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