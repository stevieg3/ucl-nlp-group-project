"""
Utilities for fine-tuning BERT base on text classification tasks, loading model and making predictions.

Some steps adapted from https://medium.com/@aniruddha.choudhury94/part-2-bert-fine-tuning-tutorial-with-pytorch-for-text
-classification-on-the-corpus-of-linguistic-18057ce330e1
"""

import numpy as np
import torch
from torch.utils.data import \
    TensorDataset, \
    DataLoader
from transformers import \
    BertForSequenceClassification, \
    AdamW, \
    BertTokenizer
from transformers.trainer_utils import \
    set_seed
from tqdm import tqdm


SST_MAX_LENGTH = 70
"""
Max length of input sequence for SST dataset
"""

SST_NUM_LABELS = 5
"""
Number of labels in SST
"""

SST_BERT_HYPERPARAMETERS = {
    'batch_size': 32,
    'learning_rate': 2e-5,
    'number_of_epochs': 2,
    'max_length': SST_MAX_LENGTH
}
"""
Selected hyperparameters for fine-tuning BERT on SST dataset
"""

AGN_MAX_LENGTH = 380
"""
Max length of input sequence for AGNews dataset
"""

AGN_NUM_LABELS = 4
"""
Number of labels in AGNews
"""

AGN_BERT_HYPERPARAMETERS = {
    'batch_size': 16,
    'learning_rate': 2e-5,
    'number_of_epochs': 2,
    'max_length': AGN_MAX_LENGTH
}
"""
Selected hyperparameters for fine-tuning BERT on AGNews dataset
"""

RANDOM_SEED = 3
"""
Random seed for model fine-tuning
"""


class PreTrainedBERT:
    """
    Class for loading pre-trained BERT model for SST or AGNews datasets
    """
    def __init__(self, device: torch.device, dataset: str, model_filepath="models"):
        """
        :param device: torch.device
        :param dataset: 'sst' or 'agn'
        :param model_filepath: Filepath containing fine-tuned-bert-base-{dataset} folder
        """
        self.device = device

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        model_filepath = model_filepath + '/' + f'fine-tuned-bert-base-{dataset}'
        self.bert = BertForSequenceClassification.from_pretrained(model_filepath)
        self.bert.to(self.device)

        if dataset == 'sst':
            self.hyperparameter_dict = SST_BERT_HYPERPARAMETERS.copy()
        elif dataset == 'agn':
            self.hyperparameter_dict = AGN_BERT_HYPERPARAMETERS.copy()

    def predict(self, sentence_array: np.array) -> (np.array, np.array):
        """
        Make predictions

        :param sentence_array: NumPy array of sentences
        :return: NumPy array of logits by class, NumPy array of probabilities by class
        """
        logits, probs = make_predictions(
            sentence_array=sentence_array,
            model=self.bert,
            tokenizer=self.tokenizer,
            device=self.device,
            hyperparameter_dict=self.hyperparameter_dict
        )

        return logits, probs


def _pad_sentence_at_end(sentence: list, max_length: int) -> np.array:
    """
    Pad tokenised sentence with zeros at end

    :param: sentence: list of encodings for a sentence
    :param: max_length: max length to pad up to
    """
    num_zeros_to_add = max_length - len(sentence)
    zero_list = list(
        np.zeros(num_zeros_to_add).astype(int)
    )
    padded_sentence = sentence + zero_list
    return np.array(padded_sentence)


def _create_sentence_input_arrays(list_encoded_sentences: list, max_length: int) -> (np.array, np.array):
    """
    Create input arrays for BERT

    :param: list_encoded_sentences: List of sentence encoding lists
    :param: max_length: max length to pad up to
    """
    encoded_sentences = [_pad_sentence_at_end(sent, max_length) for sent in list_encoded_sentences]

    train_array = np.vstack(encoded_sentences)

    train_attention_mask_array = (train_array != 0).astype(int)

    return train_array, train_attention_mask_array


def fine_tune_bert(
        device: torch.device,
        train_data_loader: DataLoader,
        dev_data_loader: DataLoader,
        num_labels: int,
        hyperparameter_dict: dict
) -> BertForSequenceClassification:
    """
    Fine tune BERT-base-uncased

    :param device: Torch device
    :param train_data_loader: DataLoader object for training data
    :param dev_data_loader: DataLoader object for development data
    :param num_labels: Number of target labels
    :param hyperparameter_dict: Dictionary of model hyperparameters
    :return: Fine-tuned BERT model
    """
    set_seed(RANDOM_SEED)

    bert_model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=num_labels,
        output_attentions=False,
        output_hidden_states=False
    )

    bert_model.to(device)

    optimizer = AdamW(
        bert_model.parameters(),
        lr=hyperparameter_dict['learning_rate']
    )

    for epoch in range(hyperparameter_dict['number_of_epochs']):

        # Training

        bert_model.train()

        for batch in tqdm(train_data_loader):

            batch_input_ids = batch[0].to(device)
            batch_attention_mask = batch[1].to(device)
            batch_labels = batch[2].to(device)

            optimizer.zero_grad()  # Set gradients to 0 otherwise will accumulate

            outputs = bert_model(
                input_ids=batch_input_ids,
                token_type_ids=None,
                attention_mask=batch_attention_mask,
                labels=batch_labels
            )

            loss = outputs[0]

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(bert_model.parameters(), 1.0)

            optimizer.step()

        # Evaluate

        bert_model.eval()

        # Train accuracy:
        train_pred_labels = []
        train_labels = []

        for batch in train_data_loader:

            batch_input_ids = batch[0].to(device)
            batch_attention_mask = batch[1].to(device)
            batch_labels = batch[2].to(device)

            with torch.no_grad():
                outputs = bert_model(
                    input_ids=batch_input_ids,
                    token_type_ids=None,
                    attention_mask=batch_attention_mask
                )

            logits = outputs[0]

            batch_pred_labels = list(
                torch.argmax(logits, dim=1).cpu().numpy()
            )
            train_pred_labels = train_pred_labels + batch_pred_labels

            batch_labels = list(
                batch_labels.cpu().numpy()
            )
            train_labels = train_labels + batch_labels

        train_accuracy = (np.array(train_pred_labels) == np.array(train_labels)).mean()

        # Dev accuracy:
        dev_pred_labels = []
        dev_labels = []

        for batch in dev_data_loader:

            batch_input_ids = batch[0].to(device)
            batch_attention_mask = batch[1].to(device)
            batch_labels = batch[2].to(device)

            with torch.no_grad():
                outputs = bert_model(
                    input_ids=batch_input_ids,
                    token_type_ids=None,
                    attention_mask=batch_attention_mask
                )

            logits = outputs[0]

            batch_pred_labels = list(
                torch.argmax(logits, dim=1).cpu().numpy()
            )
            dev_pred_labels = dev_pred_labels + batch_pred_labels

            batch_labels = list(
                batch_labels.cpu().numpy()
            )
            dev_labels = dev_labels + batch_labels

        dev_accuracy = (np.array(dev_pred_labels) == np.array(dev_labels)).mean()

        print(f"Epoch {epoch+1}: train_acc={train_accuracy}, dev_acc={dev_accuracy}")

    return bert_model


def make_predictions(
    sentence_array: np.array,
    model: BertForSequenceClassification,
    tokenizer: BertTokenizer,
    device: torch.device,
    hyperparameter_dict: dict
) -> (np.array, np.array):
    """
    Make predictions on DataFrame containing sentences with given model

    :param model: Torch model
    :param tokenizer: BERT-base tokenizer
    :param device: Torch device
    :param max_length: Max length of input sequence (for padding)
    :param hyperparameter_dict: Dictionary of model hyperparameters
    :return: NumPy array of label predictions
    """
    # Prepare data
    encoded_sentences = []

    for sentence in sentence_array:
        enc_sent_as_list = tokenizer.encode(sentence, add_special_tokens=True)
        encoded_sentences.append(enc_sent_as_list)

    input_array, input_attention_mask_array = _create_sentence_input_arrays(
        encoded_sentences,
        hyperparameter_dict['max_length']
    )

    input_tensor = torch.tensor(input_array)
    input_attention_mask_tensor = torch.tensor(input_attention_mask_array)

    input_dataset = TensorDataset(input_tensor, input_attention_mask_tensor)

    input_data_loader = DataLoader(input_dataset, batch_size=hyperparameter_dict['batch_size'])

    # Run model

    model.eval()

    logit_list = []

    for batch in tqdm(input_data_loader):

        batch_input_ids = batch[0].to(device)
        batch_attention_mask = batch[1].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=batch_input_ids,
                token_type_ids=None,
                attention_mask=batch_attention_mask
            )

        logits = outputs[0]
        logit_list.append(logits)

    logits_tensor = torch.cat(logit_list, dim=0)
    prob_tensor = torch.softmax(logits_tensor, dim=1)

    return np.array(logits_tensor), np.array(prob_tensor)
