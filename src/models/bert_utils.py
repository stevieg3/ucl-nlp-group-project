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
    AdamW
from transformers.trainer_utils import \
    set_seed
from tqdm import tqdm


MAX_LENGTH = 70
"""
Max length of input sequence
"""

BERT_HYPERPARAMETERS = {
    'batch_size': 32,
    'learning_rate': 2e-5,
    'number_of_epochs': 2
}

RANDOM_SEED = 3
"""
Random seed for model fine-tuning
"""


def pad_sentence_at_end(sentence, max_length):
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


def create_sentence_input_arrays(list_encoded_sentences, max_length):
    """
    Create input arrays for BERT

    :param: list_encoded_sentences: List of sentence encoding lists
    :param: max_length: max length to pad up to
    """
    encoded_sentences = [pad_sentence_at_end(sent, max_length) for sent in list_encoded_sentences]

    train_array = np.vstack(encoded_sentences)

    train_attention_mask_array = (train_array != 0).astype(int)

    return train_array, train_attention_mask_array


def fine_tune_bert(device, train_data_loader, dev_data_loader):
    """
    Fine tune BERT-base-uncased

    :param device: Torch device
    :param train_data_loader: DataLoader object for training data
    :param dev_data_loader: DataLoader object for development data
    :return: Fine-tuned BERT model
    """
    bert_model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=5,
        output_attentions=False,
        output_hidden_states=False
    )

    bert_model.to(device)

    optimizer = AdamW(
        bert_model.parameters(),
        lr=BERT_HYPERPARAMETERS['learning_rate']
    )

    set_seed(RANDOM_SEED)

    for epoch in range(BERT_HYPERPARAMETERS['number_of_epochs']):

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


def make_predictions(df, model, tokenizer, sentence_col_name, device):
    """
    Make predictions on DataFrame containing sentences with given model

    :param df: DataFrame containing input sentences to be classified
    :param model: Torch model
    :param tokenizer: BERT-base tokenizer
    :param sentence_col_name: Name of column containing input sentences
    :param device: Torch device
    :return: NumPy array of label predictions
    """
    # Prepare data

    df = df.copy()

    encoded_sentences = []

    for sentence in df[sentence_col_name].values:
        enc_sent_as_list = tokenizer.encode(sentence, add_special_tokens=True)
        encoded_sentences.append(enc_sent_as_list)

    input_array, input_attention_mask_array = create_sentence_input_arrays(
        encoded_sentences,
        MAX_LENGTH
    )

    input_tensor = torch.tensor(input_array)
    input_attention_mask_tensor = torch.tensor(input_attention_mask_array)

    input_dataset = TensorDataset(input_tensor, input_attention_mask_tensor)

    input_data_loader = DataLoader(input_dataset, batch_size=32)

    # Run model

    model.eval()

    predicted_labels = []

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

        batch_pred_labels = list(
            torch.argmax(logits, dim=1).cpu().numpy()
        )
        predicted_labels = predicted_labels + batch_pred_labels

    return np.array(predicted_labels)
