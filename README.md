# [COLAB DEMO](https://colab.research.google.com/drive/1tdj3qxcqNTxWubZ7RpjM2EiCYNIcGe-3?usp=sharing)

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

Download and unzip the fine-tuned models into `models/` directory. The links are given below:

* [bcn-agnews_output.zip](https://liveuclac-my.sharepoint.com/:u:/g/personal/ucabkro_ucl_ac_uk/ESZ14c-PS39BuIt7YXqnrL4Br368_v8cjo6X7GMPt-9N3A?e=TYs9A1)
* [bcn-sst_output.zip](https://liveuclac-my.sharepoint.com/:u:/g/personal/ucabkro_ucl_ac_uk/EbhqABBIa5BHsLBE-LuUYzcBykHKxAwMJofFVyZiLrZoHQ?e=Vq8l4x)
* [bert-agnews.zip](https://liveuclac-my.sharepoint.com/:u:/g/personal/ucabkro_ucl_ac_uk/Ea6aDI5-1xFEje2Olvj7KmQBq6FLglaW_2Eoez7vB82_Ow?e=Myuhk0)
* [bert-sst.zip](https://liveuclac-my.sharepoint.com/:u:/g/personal/ucabkro_ucl_ac_uk/EZlG-PDyYbpEgb1Xyg_tiTUBUrIvPYrvh25mnr2R1r689g?e=b1PqPf)

## Datasets

All of the datasets are cached on the running system by the [pytreebank](https://github.com/JonathanRaiman/pytreebank) and [huggingface datasets](https://github.com/huggingface/datasets) libraries.
