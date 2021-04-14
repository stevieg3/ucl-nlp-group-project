#!/usr/bin/env python3


import os
import sys
import subprocess
import json
import requests
import numpy as np
import pandas as pd
import typing
from overrides import overrides
from abc import abstractmethod

import tagging
import spacy
import allennlp
import allennlp.predictors

import torch
import torch.utils
import torch.utils.data
import transformers

# add project root directory to the search path
project_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root_dir)
from src.data.dataload import *
import notebooks.AllenNLP.BCN_model as BCN_model


class Model:
    def __init__(self):
        self.model = None

    def _download_if_not_exists(self, url: str, filepath: str, override=False) -> str:
        if os.path.isfile(filepath) and not override:
            print('downloading into', filepath)
            with open(filepath, 'wb') as f:
                response = requests.get(url)
                f.write(response.content)
            print('download finished')
        return filepath

    def _get_model_url_for_dataset(self, dataset) -> typing.Optional[str]:
        return None

    @abstractmethod
    def _get_model_filepath_for_dataset(self, dataset) -> str:
        pass

    @abstractmethod
    def _load_finetuned_model(self, filepath: str) -> None:
        pass

    def _load_model_from_url(self, url: str, filepath: str, override=False) -> None:
        self._download_if_not_exists(url=url, filepath=filepath, override=override)
        self._load_finetuned_model(filepath)

    @abstractmethod
    def _finetune_for_dataset(self, dataset, filepath: str) -> None:
        pass

    def _load_model_from_dataset(self, dataset, override=False) -> None:
        filepath = self._get_model_filepath_for_dataset(dataset)
        url = self._get_model_url_for_dataset(dataset)
        if url is None:
            if not os.path.exists(filepath) or override:
                self._finetune_for_dataset(dataset=dataset, filepath=filepath)
            self._load_finetuned_model(filepath=filepath)
        else:
            self._load_model_from_url(url=url, filepath=filepath, override=override)

    def load_model(self, dataset) -> None:
        self._load_model_from_dataset(dataset)


@allennlp.predictors.predictor.Predictor.register('allennlp_text_classifier')
class AllenNLPClassifier(allennlp.predictors.predictor.Predictor):
    """
    Predictor for any model that takes in a sentence and returns
    a single class for it.  In particular, it can be used with
    the [`BasicClassifier`](../models/basic_classifier.md) model.

    """

    def predict(self, sentence: str) -> allennlp.common.util.JsonDict:
        return self.predict_json({Dataset.SENTENCE: sentence})

    @overrides
    def _json_to_instance(self, json_dict: allennlp.common.util.JsonDict) -> allennlp.data.Instance:
        """
        Expects JSON that looks like `{"sentence": "..."}`.
        Runs the underlying model, and adds the `"label"` to the output.
        """
        sentence = json_dict[Dataset.SENTENCE]
        reader_has_tokenizer = (
            getattr(self._dataset_reader, "tokenizer", None) is not None
            or getattr(self._dataset_reader, "_tokenizer", None) is not None
        )
        if not reader_has_tokenizer:
            tokenizer = allennlp.data.tokenizers.spacy_tokenizer.SpacyTokenizer()
            sentence = tokenizer.tokenize(sentence)
        return self._dataset_reader.text_to_instance(sentence)

    @overrides
    def predictions_to_labeled_instances(self,
            instance: allennlp.data.Instance,
            outputs: typing.Dict[str, np.ndarray]) \
                -> typing.List[allennlp.data.Instance]:
        new_instance = instance.duplicate()
        label = np.argmax(outputs["class_probabilities"])
        new_instance.add_field(Dataset.TARGET, allennlp.data.fields.LabelField(int(label), skip_indexing=True))
        return [new_instance]



class BCNModel(Model):
    def __init__(self, cache_dir='', config_file_template=None):
        self.nlp = spacy.load('en_core_web_sm')
        self.model = None
        self.vocab = None
        self.predictor = None
        self.cache_dir = cache_dir
        self.output_dir = ''
        if config_file_template is None:
            config_file_template = 'config_BCN.jsonnet.template'
        self.config_file_template = config_file_template
        self.config_file = None

    def _get_model_filepath_for_dataset(self, dataset) -> str:
        self.output_dir = os.path.join(self.cache_dir, f'bcn-{dataset.NAME}_output')
        self.config_file = os.path.join(self.cache_dir, f'config_BCN_{dataset.NAME}.jsonnet')
        return os.path.join(self.output_dir, 'model.tar.gz')

    def _finetune_setup_config(self, dataset):
        config_bcn = None
        with open(self.config_file_template, 'r') as f:
            config_bcn = json.load(f)
        trainfile, valfile, testfile = dataset.save_train_val_test_jsonl(dirname=self.cache_dir)
        print('saved', [trainfile, valfile, testfile])
        config_bcn['dataset_reader'] = {
            'type': 'allennlp_reader'
        }
        config_bcn['validation_dataset_reader'] = {
            'type': 'allennlp_reader'
        }
        config_bcn['train_data_path'] = trainfile
        config_bcn['validation_data_path'] = valfile
        with open(self.config_file, 'w') as f:
            json.dump(config_bcn, f)
        print('config file', self.config_file)

    def _finetune_for_dataset(self, dataset, filepath: str) -> None:
        self._finetune_setup_config(dataset)
        command = ['allennlp', 'train']
        command += ['--include-package', 'tagging']
        command += ['-s', self.output_dir]
        command += [self.config_file]
        print('executing', command)
        subprocess.call(command)
        assert os.path.isdir(self.output_dir)

    def _load_finetuned_model(self, filepath: str) -> None:
        assert os.path.isfile(filepath), f'file "{filepath}" does not exist'
        archive = allennlp.models.archival.load_archive(filepath)
        self.model = archive.model
        self.vocab = self.model.vocab
        self.predictor = allennlp.predictors.predictor.Predictor.from_archive(archive, 'allennlp_text_classifier')

    def predict(self, s: str):
        return self.predictor.predict(sentence=s)


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

    def _load_finetuned_model(self, filepath: str) -> None:
        if os.path.isdir(self.output_dir) and not os.path.isfile(filepath):
            print('[ATTENTION]', f'please remove "{self.output_dir}" and try again')
        assert os.path.isfile(filepath), f'file "{filepath}" does not exist'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = transformers.BertForSequenceClassification \
                                 .from_finetuned(filepath)
        self.model.to(self.device)

    def _load_tokenizer(self, do_lower_case=True, **kwargs) -> None:
        self.tokenizer = transformers.BertTokenizer \
                                     .from_finetuned(self.tokenizer_type,
                                                      do_lower_case=do_lower_case,
                                                      **kwargs)

    def load_tokenizer(self, **kwargs) -> None:
        self._load_tokenizer(**kwargs)

    # TODO
    def predict(self, s: str):
        pass


if __name__ == "__main__":
    import bpython
    agnews = load_agnews()
    bcn = BCNModel()
    bcn.load_model(agnews)
    bpython.embed(locals_=dict(globals(), **locals()))
#    sst = load_sst()
#    bert = BERTModel(bert_type=BERT_BASE,
#                     tokenizer_type=TOKENIZER_UNCASED)
#    bert.load_model(sst)
#    bert.load_tokenizer()
#    bpython.embed(locals_=dict(globals(), **locals()))
