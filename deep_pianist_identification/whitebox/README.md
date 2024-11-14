# White-box classifiers

These scripts reproduce the figures and results obtained from the "white-box" models described in section 3 of our
paper.

## Fitting and optimizing models

We have created a command line script that makes it easy to fit and optimize each of the model types we evaluate. To run
this, make sure you have all the data available locally (see `preprocessing`) and have set up the environment (
recommended way is by running `pip install -r requirements.txt` inside a `virtualenv`). Then run the following script
from the root directory of the project:

```
python deep_pianist_identification/whitebox/create_classifier.py -c <classifier-type>
```

where `<classifier-type>` is a string in `["lr", "rf", "svm"]`: i.e., logistic regression (default), random forest,
support vector machine.

Without any arguments, this will do the following:

- Extract chords and n-grams from the data, using the settings outlined in our paper
- Fit the provided classifier type to the training split and optimize for 10,000 iterations against the validation split
    - The optimized parameter settings used to create the results in the paper are loaded by default for each
      classifier.
    - To optimize from scratch, remove or rename the `.csv` files inside `./references/whitebox`. Note that optimization
      will take a very long time on weaker systems, or may not finish.
    - Multiprocessing is leveraged by default to speed optimization up and is controlled by the `N_JOBS` parameter
      inside `wb_utils.py`.
- Get the best hyperparameters and report accuracy and top-k accuracy from predicting the held-out test split
    - Note that top-k accuracy using SVM is not likely to be accurate due to constraints in the `sklearn`
      implementation: for more information, [see this thread](https://github.com/scikit-learn/scikit-learn/issues/13211)
- Compute permutation feature importance (results stored in `./reports/figures/whitebox/permutation`)

When `<classifier-type>` == `lr`, two additional outputs are created by:

- Extract top- and bottom-K weights from the model for each performer across both harmony and melody and create plots (
  in `.reports/figures/whitebox/lr_weights`)
- Compute the correlation between top-K weights from the full model using models fitted to both JTD and PiJAMA
  separately (results stored in `./reports/figures/whitebox/database`)

## Script arguments

We allow for several arguments to be passed in to the command line script.

- `-d` / `--dataset`: the name of a folder containing data split folders inside `./references/data_split/`. Defaults to
  `20class_80min`, which are the splits used in the paper.
- `-i` / `--n-iter`: number of iterations to use in optimizing the model and when bootstrapping parameter settings
- `-l` / `--ngrams`: the _n_-gram settings to use. Defaults to `[2, 3]` as is reported in the paper.
- `-s` / `--min-count`: the minimum number of tracks a feature must appear in to be used. Defaults to 10 as in the
  paper.
- `-b` / `--max-count`: the maximum number of tracks a feature can appear in before it is dropped. Defaults to 1000 as
  in the paper.
- `-c` / `--classifier-type`: a string corresponding to a given model type. Either `rf` (random forest), `lr` (logistic
  regression, default), `svm` (support vector machine) are recognised. Note that not all outputs will be created for all
  model types.
- `-z` / `--scale`: whether to scale data using z-transformation. Defaults to `True`.

Additional parameter settings can be changed by altering various constances (defined in CAPITAL_LETTERS) in scripts such
as `wb_utils.py`, `features.py`. However, be warned that this may break things!