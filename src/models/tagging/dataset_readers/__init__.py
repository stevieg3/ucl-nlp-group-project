"""
A :class:`~allennlp.data.dataset_readers.dataset_reader.DatasetReader`
reads a file and converts it to a collection of
:class:`~allennlp.data.instance.Instance` s.
The various subclasses know how to read specific filetypes
and produce datasets in the formats required by specific models.
"""

import os
import sys

project_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root_dir not in sys.path:
    sys.path.append(project_root_dir)

from allennlp.data.dataset_readers.dataset_reader import (
    DatasetReader,
    WorkerInfo,
    DatasetReaderInput,
)
from allennlp.data.dataset_readers.babi import BabiReader
from allennlp.data.dataset_readers.conll2003 import Conll2003DatasetReader
from allennlp.data.dataset_readers.interleaving_dataset_reader import InterleavingDatasetReader
from allennlp.data.dataset_readers.multitask import MultiTaskDatasetReader
from allennlp.data.dataset_readers.sequence_tagging import SequenceTaggingDatasetReader
from allennlp.data.dataset_readers.sharded_dataset_reader import ShardedDatasetReader
from allennlp.data.dataset_readers.text_classification_json import TextClassificationJsonReader
from src.models.tagging.dataset_readers.allennlp_reader import AllenNLPReader
