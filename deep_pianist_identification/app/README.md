# Generating webapp content

The scripts in this folder will generate the content used in the webapp hosted
at [this link](https://huwcheston.github.io/ImprovID-app/index.html). You only need to run them if you want to
regenerate the web application yourself, so these instructions are included mostly for completeness.

First, clone the repo, install the requirements, and download the data (see the main README.md file).

To build the "melody" pages, you'll need to fit the classifiers by running `whitebox/create_classifier.py` with the
argument `-c lr`. Then, you can create the HTML pages by running `app/generate_melody_pages.py`.

To build the "harmony" pages, you'll need to ensure that you've downloaded the trained checkpoints, then create the
concept activation vectors by running `explainability/create_harmony_cavs.py`. You can generate the HTML by running
`app/generate_harmony_pages.py`.

Finally, to build the "style" pages, you'll need to download the checkpoints and run `validation.py` using the model.
You can then build the HTML by running `app/generate_style_pages.py`.
