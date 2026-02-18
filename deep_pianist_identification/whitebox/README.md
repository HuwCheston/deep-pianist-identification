# Handcrafted features models

This README walks through the process for training and generating outputs from the "handcrafted feature" models
described in section 3 of our paper.

## Fitting and optimizing models

First, make sure you've followed the *setup* section on [this README](../../README.md)

We have created a command line script that makes it easy to fit and optimize each of the model types we evaluate in the
paper. Run the following script from the root directory of the project:

```
python deep_pianist_identification/whitebox/create_classifier.py -c <classifier-type> -l <n1-size> <n2-size> <n3-size> ...
```

where `<classifier-type>` is a string in `["lr", "rf", "svm"]`: i.e., logistic regression (default), random forest,
support vector machine, and `-l` is a sequence of integers defining the size of features that should be extracted.

Without any arguments, this will do the following:

- Extract chords and n-grams from the data, using the settings outlined in our paper
- Fit the provided classifier type to the training split and optimize for 1,000 iterations against the validation split
    - The optimized parameter settings used to create the results in the paper are loaded by default for each
      classifier.
    - To optimize from scratch, remove or rename the `.csv` files inside `./references/whitebox`. Note that optimization
      will take a very long time on weaker systems, or may not finish.
    - Multiprocessing is leveraged by default to speed optimization up and is controlled by the `N_JOBS` parameter
      inside `wb_utils.py`.
- Get the best hyperparameters and report accuracy and top-k accuracy from predicting the held-out test split
    - Note that top-k accuracy using SVM is not likely to be accurate due to constraints in the `sklearn`
      implementation: for more information, [see this thread](https://github.com/scikit-learn/scikit-learn/issues/13211)
- Compute outputs (results stored in `./reports/figures/whitebox/...`).

The following outputs are computed for diferent model types:

- `<classifier-type>` == `lr`, `rf`, `svm`:
    - Permutation feature importance (`./reports/figures/whitebox/permutation`)
    - PCA decomposition of 4-grams (`./reports/figures/whitebox/pca_feature_counts`)
    - Counts for all features (`./reports/figures/whitebox/barplot_feature_counts.png`)
- `<classifier-type>` == `lr`
    - Correlations between features from both datasets (`./reports/figures/whitebox/database`)
    - Weights for top- and bottom-k features for each performer (`./reports/figures/whitebox/lr_weights`)

## Script arguments

We allow for several arguments to be passed in to the command line script.

<details>
<summary>View arguments</summary>

- `-d` / `--dataset`: the name of a folder containing data split folders inside `./references/data_split/`. Defaults to
  `20class_80min`, which are the splits used in the paper.
- `-i` / `--n-iter`: number of iterations to use in optimizing the model and when bootstrapping parameter settings
- `-l` / `--feature-sizes`: the "size" of the features to extract, *expressed in intervals*. Defaults to `2 3 4 5 6`
  which
  corresponds to chords and n-grams containing either three to seven pitches. Pass in
- `-s` / `--min-count`: the minimum number of tracks a feature must appear in to be used. Defaults to 10 as in the
  paper.
- `-b` / `--max-count`: the maximum number of tracks a feature can appear in before it is dropped. Defaults to 1000 as
  in the paper.
- `-c` / `--classifier-type`: a string corresponding to a given model type. Either `rf` (random forest), `lr` (logistic
  regression, default), `svm` (support vector machine) are recognised. Note that not all outputs will be created for all
  model types.
- `-z` / `--scale`: whether to scale data using tf-idf. Defaults to `True`.
- `-k` / `--database-k-coefs`: number of features to use when computing correlation between different datasets, default
  is `2000`. Either `int` or `float` is accepted: `float` is interpreted as fraction of total features. `-1` is
  interpreted as using all available features.
- `--optimize`: whether or not to use `optuna` to optimize the parameters for the classifier (rather than random sampling). Defaults to random sampling.

</details>

Additional parameter settings can be changed by altering various constances (defined in CAPITAL_LETTERS) in scripts such
as `wb_utils.py`, `features.py`. However, be warned that this may break things!