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

