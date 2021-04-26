#!/usr/bin/env python3


import os
import sys
project_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root_dir not in sys.path:
    sys.path.append(project_root_dir)
from src.data.dataload import *
from src.models.bert_utils import *
from src.models.model import *


class BertModel(Model):
    """
    Class for loading pre-trained BERT model for SST or AGNews datasets
    """

    MODELTYPE = 'torch'
    HYPERPARAMETERS = {
        DatasetSST.NAME: dict(
            batch_size=32,
            learning_rate=2e-5,
            number_of_epochs=2,
            max_length=70
        ),
        DatasetAGNews.NAME: dict(
            batch_size=16,
            learning_rate=2e-5,
            number_of_epochs=2,
            max_length=380
        )
    }

    def __init__(self, device=None):
        """
        :param device: torch.device
        """
        super(BertModel, self).__init__()
        if device is None:
            device = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device

    @overrides
    def _get_model_filepath_for_dataset(self, dataset) -> str:
        return self._path_to(f'fine-tuned-bert-base-{dataset.NAME}', dirname=self.modeldir)

    @overrides
    def _finetune_for_dataset(self, dataset, filepath: str) -> None:
        raise NotImplementedError()

    @overrides
    def _load_finetuned_model(self, filepath: str) -> None:
        assert os.path.isdir(filepath), f'directory "{filepath}" does not exist'
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.model = BertForSequenceClassification.from_pretrained(filepath)
        self.model.to(self.device)
        self.hyperparameter_dict = BertModel.HYPERPARAMETERS[self.dataset_finetune.NAME].copy()

    @overrides
    def predict(self, s: str) -> pd.DataFrame:
        pred = self.predict_batch(np.array([s], dtype=object))
        n_classes = len(pred.logits[0])
        return pd.DataFrame(dict(
            logits=pred.logits[0],
            class_probabilities=pred.class_probabilities[0],
            label=np.repeat(pred.label[0], repeats=n_classes)
        ))

    @overrides
    def predict_batch(self, s: typing.Iterable[str]) -> pd.DataFrame:
        logits, probs = make_predictions(
            sentence_array=s,
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            hyperparameter_dict=self.hyperparameter_dict
        )
        return pd.DataFrame(dict(
            logits=logits.tolist(),
            class_probabilities=probs.tolist(),
            label=logits.argmax(axis=1)
        ))


if __name__ == "__main__":
    import bpython
    data = load_sst()
    bert = BertModel()
    bert.load_model(data)
    bpython.embed(locals_=dict(globals(), **locals()))
