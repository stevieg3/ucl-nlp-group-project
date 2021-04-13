#!/usr/bin/env python3


import os
import requests
import pandas as pd
import typing

import torch
import torch.utils
import torch.utils.data
import transformers


class Model:
    def __init__(self):
        pass

    def _download_if_not_exists(self, url: str, filepath: str, override=False) -> str:
        if os.path.isfile(filepath) and not override:
            print('downloading into', filepath)
            with open(filepath, 'wb') as f:
                response = requests.get(url)
                f.write(response.content)
            print('download finished')
        return filepath


def BCNModel(Model):
    def __init__(self):
        pass


BERT_BASE, BERT_LARGE = 'base', 'large'
TOKENIZER_UNCASED = 'uncased'
class BERTModel(Model):

    def __init__(self, cache_dir='.', bert_type=BERT_BASE, tokenizer_type=TOKENIZER_UNCASED) -> None:
        assert bert_type in [BERT_BASE, BERT_LARGE]
        self.bert_type = bert_type
        self.tokenizer_type = f'bert-{bert_type}-{tokenizer_type}'
        self.device = None
        self.model = None
        self.tokenizer = None
        self.cache_dir = cache_dir

    def _get_model_url_for_dataset(self, dataset) -> typing.Optional[str]:
        registry: dict = {
            BERT_BASE: dict(
            ),
            BERT_LARGE: dict(
            )
        }
        if dataset.NAME not in registry:
            return None
        return registry[dataset.NAME]

    def _get_model_filepath_for_dataset(self, dataset) -> str:
        filename = f'fine-tuned-bert-{self.bert_type}-{dataset.NAME.lower()}'
        return os.path.join(self.cache_dir, filename)

    def _load_pretrained_bert(self, filepath: str) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = transformers.BertForSequenceClassification \
                                 .from_pretrained(filepath)
        self.model.to(self.device)

    def _load_model_from_url(self, url: str, filepath: str) -> None:
        self._download_if_not_exists(url=url, filepath=filepath)
        self._load_pretrained_bert(filepath)

    def _load_model_from_dataset(self, dataset, override=False) -> None:
        filepath = self._get_model_filepath_for_dataset(dataset)
        url = self._get_model_url_for_dataset(dataset)
        if url is None:
            self._load_pretrained_bert(filepath=filepath)
        else:
            self._load_from_url(url=url, filepath=filepath)

    def load_model(self, dataset) -> None:
        self._load_model_from_dataset(dataset)

    def _load_tokenizer(self, do_lower_case=True, **kwargs) -> None:
        self.tokenizer = transformers.BertTokenizer \
                                     .from_pretrained(self.tokenizer_type,
                                                      do_lower_case=do_lower_case,
                                                      **kwargs)

    def load_tokenizer(self, **kwargs) -> None:
        self._load_tokenizer(**kwargs)

    # TODO
    def predict(self):
        pass


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.abspath('../..'))
    from src.data.dataload import *
    import bpython
    sst = load_sst()
    bert = BERTModel(bert_type=BERT_BASE,
                     tokenizer_type=TOKENIZER_UNCASED)
    bert.load_model(sst)
    bert.load_tokenizer()
    bpython.embed(locals_=dict(globals(), **locals()))
