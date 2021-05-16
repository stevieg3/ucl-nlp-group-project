# About

This repository contains the code for the COMP0087 group project "On the Robustness of Natural Language Explainers".

# Structure

* src/ - code imported into the notebooks
* notebooks/ - contains jupyter notebooks of various sorts all related to the project
    * notebooks/fine-tune-models/ - code to fine-tune bert models for sst and ag news datasets
    * notebooks/code-demo/ - examples of using the code in the notebooks
    * notebooks/experiments/ - notebooks used for running experiments
* models/ - fine-tuned models (not present in the repo, see how to set up)
* scripts/ - helper scripts

# Setup

## Environment

```python
conda create -n snlp64-group-project python=3.7 -y
conda activate snlp64-group-project
pip install -r requirements.txt
```

## Models

Download and unzip the models into `models/` directory. The links are given as:

* [bcn-agnews_output.zip](https://liveuclac-my.sharepoint.com/:u:/r/personal/ucabkro_ucl_ac_uk/Documents/ucl-snlp64-group-project/bcn-agnews_output.zip?csf=1&web=1&e=nCMwss)
* [bcn-sst_output.zip](https://liveuclac-my.sharepoint.com/:u:/r/personal/ucabkro_ucl_ac_uk/Documents/ucl-snlp64-group-project/bcn-sst_output.zip?csf=1&web=1&e=a1oPNf)
* [bert-agnews.zip](https://liveuclac-my.sharepoint.com/:u:/r/personal/ucabkro_ucl_ac_uk/Documents/ucl-snlp64-group-project/fine-tuned-bert-base-agnews.zip?csf=1&web=1&e=MaBYWA)
* [bert-sst.zip](https://liveuclac-my.sharepoint.com/:u:/r/personal/ucabkro_ucl_ac_uk/Documents/ucl-snlp64-group-project/fine-tuned-bert-base-sst.zip?csf=1&web=1&e=tK1KAB)

## Datasets

All of the datasets are cached on the running system by the [pytreebank](https://github.com/JonathanRaiman/pytreebank) and [huggingface datasets](https://github.com/huggingface/datasets) libraries.