import numpy as np
import pandas as pd
import spacy
import random
from copy import deepcopy

from checklist.perturb import Perturb
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer

from src.data.constants import \
    ADJECTIVES, \
    REVERSE_CONTRACTION_MAP, \
    CONTRACTION_MAP, \
    PUNCTUATION, \
    DICT_GENDER

nlp = spacy.load('en_core_web_sm')


def _custom_remove_char(text_orig, char):
    """Removes characters from a list of string
    Inputs: 
    text: String or list that has strings as elements. Transformation will be applied to all strings.
    char: character(s) to be removed. This can be a string, or a list of strings (if multiple characters 
    need to be removed)
    
    Output: modified string, or list of modified strings
    """
    if type(text_orig) == str:
        text = [text_orig]
    else:
        text = text_orig
    if type(char) == str:
        char = [char]
    result = []
    dummy_string = ''
    dummy_dict = dict.fromkeys(char, '')
    table = dummy_string.maketrans(dummy_dict)
    for i in range(len(text)):
        result.append(text[i].translate(table))
    if type(text_orig) == str:
        result = result[0]
    return result


def _gen_empty_columns():
    new_column_tokens = []
    new_column_concat = []
    new_column_success_flag = []
    new_column_pert_indices = []
    return new_column_tokens, new_column_concat, new_column_success_flag, new_column_pert_indices


def swap_adjectives(df: pd.DataFrame, orig_index: int, tokens_orig: list):
    """
    swap two consecutive adjectives (connected by 'and' or 'or')

    :param df: DataFrame containing sentences
    :param orig_index: integer indicating the column index of the original sentence
    :param tokens_orig: tokenised version of sentence
    :return: None. Modifies DataFrame in-place
    """

    new_column_tokens, new_column_concat, new_column_success_flag, new_column_pert_indices = _gen_empty_columns()

    for s in range(len(tokens_orig)):
        sentence = deepcopy(tokens_orig[s])
        pert_indices = []
        for t in range(1, len(sentence)):
            if sentence[t] in ['and', 'or']:
                if sentence[t - 1] in ADJECTIVES and sentence[t + 1] in ADJECTIVES:
                    adj_1 = str(sentence[t - 1])
                    adj_2 = str(sentence[t + 1])
                    sentence[t - 1] = adj_2
                    sentence[t + 1] = adj_1
                    pert_indices.extend([t - 1, t + 1])
        new_column_tokens.append(sentence)
        if len(pert_indices) == 0:
            new_column_concat.append(df.iloc[s, orig_index])
            new_column_success_flag.append(0)
            new_column_pert_indices.append(None)
        else:
            new_column_concat.append(" ".join(sentence))
            new_column_success_flag.append(1)
            new_column_pert_indices.append(pert_indices)

    df['swap_adj_concat'] = new_column_concat
    df['swap_adj_tokens'] = new_column_tokens
    df['swap_adj_success'] = new_column_success_flag
    df['swap_adj_pert_ind'] = new_column_pert_indices


def contraction(df: pd.DataFrame, orig_index: int, tokens_orig: list):
    """
    Change to and from contracted form (e.g. "you're" to "you are", or "you are" to "you're")

    :param df: DataFrame containing sentences
    :param orig_index: integer indicating the column index of the original sentence
    :param tokens_orig: tokenised version of sentence
    :return: None. Modifies DataFrame in-place
    """

    new_column_tokens, new_column_concat, new_column_success_flag, new_column_pert_indices = _gen_empty_columns()

    for s in range(len(tokens_orig)):
        sentence = deepcopy(tokens_orig[s])
        pert_indices = []
        t = 0
        while t < len(sentence) - 2:
            n_grams_with_space = {}
            n_grams_no_space = {}
            change_flag = 0
            for i in range(1, 3 + 1):
                n_grams_no_space[i] = "".join(sentence[t:t + i])
                n_grams_with_space[i] = " ".join(sentence[t:t + i])
            for n in range(1, 3 + 1):
                n_gram_no_space = n_grams_no_space[n]
                n_gram_with_space = n_grams_with_space[n]
                if n_gram_no_space in REVERSE_CONTRACTION_MAP:
                    new_phrase = REVERSE_CONTRACTION_MAP[n_gram_no_space]
                    # if number of tokens before and after modification is not the same, leave sentence unchanged
                    if len(new_phrase) != n:
                        continue
                    else:
                        for i in range(n):
                            sentence[t + i] = new_phrase[i]
                    for i in range(n):
                        pert_indices.append(t + i)
                    t += n
                    change_flag += 1
                    break
                elif n_gram_with_space in CONTRACTION_MAP:
                    new_phrase = CONTRACTION_MAP[n_gram_with_space]
                    # if number of tokens before and after modification is not the same, leave sentence unchanged
                    if len(new_phrase) != n:
                        continue
                    else:
                        for i in range(n):
                            sentence[t + i] = new_phrase[i]
                    for i in range(n):
                        pert_indices.append(t + i)
                    t += n
                    change_flag += 1
                    break
            # if no change has been applied to any of the n-grams, move to next token in sentence
            if change_flag == 0:
                t += 1

        if len(pert_indices) == 0:
            new_column_tokens.append(sentence)
            new_column_concat.append(df.iloc[s, orig_index])
            new_column_success_flag.append(0)
            new_column_pert_indices.append(None)
        else:
            new_column_tokens.append(sentence)
            new_column_concat.append(" ".join(sentence))
            new_column_success_flag.append(1)
            new_column_pert_indices.append(pert_indices)

    df['contraction_concat'] = new_column_concat
    df['contraction_tokens'] = new_column_tokens
    df['contraction_success'] = new_column_success_flag
    df['contraction_pert_ind'] = new_column_pert_indices


def change_first_name(df: pd.DataFrame, orig_index: int, tokens_orig: list):
    """
    Change first name in sentence if one exists and if name is in CheckList's name lookup json

    :param df: DataFrame containing sentences
    :param orig_index: integer indicating the column index of the original sentence
    :param tokens_orig: tokenised version of sentence
    :return: None. Modifies DataFrame in-place
    """

    new_column_tokens, new_column_concat, new_column_success_flag, new_column_pert_indices = _gen_empty_columns()
    # Checklist requires pre-processing with Spacy for this perturbation
    pdata = list(nlp.pipe(df.iloc[:, orig_index]))

    for s in range(len(tokens_orig)):
        sentence = tokens_orig[s]
        new_sentence = Perturb.change_names(pdata[s], n=1, first_only=True, meta=True)
        if not new_sentence:
            new_column_tokens.append(sentence)
            new_column_concat.append(df.iloc[s, orig_index])
            new_column_success_flag.append(0)
            new_column_pert_indices.append(None)
        else:
            # extract token that has been perturbed
            token_pert = new_sentence[1][0][0]
            # verify that perturbed name appears as token in the original input
            if token_pert in sentence:
                # obtain index
                token_index = sentence.index(token_pert)
                new_sentence_tokens = deepcopy(sentence)
                # replace token
                new_sentence_tokens[token_index] = new_sentence[1][0][1]
                # update columns
                new_column_tokens.append(new_sentence_tokens)
                new_column_concat.append(new_sentence[0][0])
                new_column_success_flag.append(1)
                new_column_pert_indices.append([token_index])
            # if token cannot be found in original list of tokens
            else:
                new_column_tokens.append(sentence)
                new_column_concat.append(df.iloc[s, orig_index])
                new_column_success_flag.append(0)
                new_column_pert_indices.append(None)

    df['change_first_name_concat'] = new_column_concat
    df['change_first_name_tokens'] = new_column_tokens
    df['change_first_name_success'] = new_column_success_flag
    df['change_first_name_pert_ind'] = new_column_pert_indices


def change_last_name(df: pd.DataFrame, orig_index: int, tokens_orig: list):
    """
    Change last name in sentence if one exists and if name is in CheckList's name lookup json

    :param df: DataFrame containing sentences
    :param orig_index: integer indicating the column index of the original sentence
    :param tokens_orig: tokenised version of sentence
    :return: None. Modifies DataFrame in-place
    """

    new_column_tokens, new_column_concat, new_column_success_flag, new_column_pert_indices = _gen_empty_columns()
    # Checklist requires pre-processing with Spacy for this perturbation
    pdata = list(nlp.pipe(df.iloc[:, orig_index]))

    for s in range(len(tokens_orig)):
        sentence = tokens_orig[s]
        new_sentence = Perturb.change_names(pdata[s], n=1, last_only=True, meta=True)
        if not new_sentence:
            new_column_tokens.append(sentence)
            new_column_concat.append(df.iloc[s, orig_index])
            new_column_success_flag.append(0)
            new_column_pert_indices.append(None)
        else:
            # extract token that has been perturbed
            token_pert = new_sentence[1][0][0]
            # verify that perturbed name appears as token in the original input
            if token_pert in sentence:
                # obtain index
                token_index = sentence.index(token_pert)
                new_sentence_tokens = deepcopy(sentence)
                new_sentence_tokens[token_index] = new_sentence[1][0][1]
                new_column_tokens.append(new_sentence_tokens)
                new_column_concat.append(new_sentence[0][0])
                new_column_success_flag.append(1)
                new_column_pert_indices.append([token_index])
            else:
                new_column_tokens.append(sentence)
                new_column_concat.append(df.iloc[s, orig_index])
                new_column_success_flag.append(0)
                new_column_pert_indices.append(None)

    df['change_last_name_concat'] = new_column_concat
    df['change_last_name_tokens'] = new_column_tokens
    df['change_last_name_success'] = new_column_success_flag
    df['change_last_name_pert_ind'] = new_column_pert_indices


def change_location(df: pd.DataFrame, orig_index: int, tokens_orig: list):
    """
    Change location in sentence if one exists and if location is in CheckList's location lookup json

    :param df: DataFrame containing sentences
    :param orig_index: integer indicating the column index of the original sentence
    :param tokens_orig: tokenised version of sentence
    :return: None. Modifies DataFrame in-place
    """

    new_column_tokens, new_column_concat, new_column_success_flag, new_column_pert_indices = _gen_empty_columns()
    # Checklist requires pre-processing with Spacy for this perturbation
    pdata = list(nlp.pipe(df.iloc[:, orig_index]))

    for s in range(len(tokens_orig)):
        sentence = tokens_orig[s]
        new_sentence = Perturb.change_location(pdata[s], n=1, meta=True)
        if not new_sentence:
            new_column_tokens.append(sentence)
            new_column_concat.append(df.iloc[s, orig_index])
            new_column_success_flag.append(0)
            new_column_pert_indices.append(None)
        else:
            # extract token that has been perturbed
            token_pert = new_sentence[1][0][0]
            # verify that perturbed name appears as token in the original input
            if token_pert in sentence:
                # obtain index
                token_index = sentence.index(token_pert)
                new_sentence_tokens = deepcopy(sentence)
                new_sentence_tokens[token_index] = new_sentence[1][0][1]
                new_column_tokens.append(new_sentence_tokens)
                new_column_concat.append(new_sentence[0][0])
                new_column_success_flag.append(1)
                new_column_pert_indices.append([token_index])
            # if token cannot be found in original list of tokens
            else:
                new_column_tokens.append(sentence)
                new_column_concat.append(df.iloc[s, orig_index])
                new_column_success_flag.append(0)
                new_column_pert_indices.append(None)

    df['change_location_concat'] = new_column_concat
    df['change_location_tokens'] = new_column_tokens
    df['change_location_success'] = new_column_success_flag
    df['change_location_pert_ind'] = new_column_pert_indices


def add_typo(df: pd.DataFrame, orig_index: int, tokens_orig: list):
    """
    Add typo using checklist library

    :param df: DataFrame containing sentences
    :param orig_index: integer indicating the column index of the original sentence
    :param tokens_orig: tokenised version of sentence
    :return: None. Modifies DataFrame in-place
    """

    new_column_tokens, new_column_concat, new_column_success_flag, new_column_pert_indices = _gen_empty_columns()

    for s in range(len(tokens_orig)):
        sentence = deepcopy(tokens_orig[s])
        if len(sentence) < 3:
            new_column_tokens.append(sentence)
            new_column_concat.append(df.iloc[s, orig_index])
            new_column_success_flag.append(0)
            new_column_pert_indices.append(None)
            continue
        token_length = 1
        attempts = 0
        while token_length <= 1 and attempts < 10:
            index_pert = random.randint(1, len(sentence) - 2)
            token_length = len(sentence[index_pert])
            attempts += 1
        if token_length <= 1:
            new_column_tokens.append(sentence)
            new_column_concat.append(df.iloc[s, orig_index])
            new_column_success_flag.append(0)
            new_column_pert_indices.append(None)
        else:
            sentence[index_pert + 1] = sentence[index_pert][-1] + sentence[index_pert + 1]
            sentence[index_pert] = sentence[index_pert][:-1]
            new_column_tokens.append(sentence)
            new_column_concat.append(" ".join(sentence))
            new_column_success_flag.append(1)
            new_column_pert_indices.append([index_pert, index_pert + 1])

    df['add_typo_concat'] = new_column_concat
    df['add_typo_tokens'] = new_column_tokens
    df['add_typo_success'] = new_column_success_flag
    df['add_typo_pert_ind'] = new_column_pert_indices


def strip_trailing_punct(df: pd.DataFrame, orig_index: int, tokens_orig: list):
    """
    Remove punctuation at end of sentence

    :param df: DataFrame containing sentences
    :param orig_index: integer indicating the column index of the original sentence
    :param tokens_orig: tokenised version of sentence
    :return: None. Modifies DataFrame in-place
    """

    new_column_tokens, new_column_concat, new_column_success_flag, new_column_pert_indices = _gen_empty_columns()

    for s in range(len(tokens_orig)):
        sentence = deepcopy(tokens_orig[s])
        if sentence[-1] in PUNCTUATION:
            sentence[-1] = ''
            new_column_concat.append(" ".join(sentence))
            new_column_success_flag.append(1)
            new_column_pert_indices.append([len(sentence)-1])
        else:
            new_column_concat.append(df.iloc[s, orig_index])
            new_column_success_flag.append(0)
            new_column_pert_indices.append(None)
        new_column_tokens.append(sentence)

    df['strip_punct_concat'] = new_column_concat
    df['strip_punct_tokens'] = new_column_tokens
    df['strip_punct_success'] = new_column_success_flag
    df['strip_punct_pert_ind'] = new_column_pert_indices


def remove_commas(df: pd.DataFrame, orig_index: int, tokens_orig: list):
    """
    Remove all commas from sentence

    :param df: DataFrame containing sentences
    :param orig_index: integer indicating the column index of the original sentence
    :param tokens_orig: tokenised version of sentence
    :return: None. Modifies DataFrame in-place
    """

    new_column_tokens, new_column_concat, new_column_success_flag, new_column_pert_indices = _gen_empty_columns()

    for s in range(len(tokens_orig)):
        sentence = tokens_orig[s]
        sentence_pert = _custom_remove_char(sentence, ',')
        if sentence == sentence_pert:
            new_column_tokens.append(tokens_orig[s])
            new_column_concat.append(df.iloc[s, orig_index])
            new_column_success_flag.append(0)
            new_column_pert_indices.append(None)
        else:
            empty_indices = []
            # tracking the indices of tokens that have been removed
            for t in range(len(sentence)):
                if sentence_pert[t] == '':
                    empty_indices.append(t)
            new_column_tokens.append(sentence_pert)
            new_column_concat.append(" ".join(sentence_pert))
            new_column_success_flag.append(1)
            new_column_pert_indices.append(empty_indices)

    df['remove_commas_concat'] = new_column_concat
    df['remove_commas_tokens'] = new_column_tokens
    df['remove_commas_success'] = new_column_success_flag
    df['remove_commas_pert_ind'] = new_column_pert_indices


def remove_all_punctuation(df: pd.DataFrame, orig_index: int, tokens_orig: list):
    """
    Remove all punctuation from sentence

    :param df: DataFrame containing sentences
    :param orig_index: integer indicating the column index of the original sentence
    :param tokens_orig: tokenised version of sentence
    :return: None. Modifies DataFrame in-place
    """

    new_column_tokens, new_column_concat, new_column_success_flag, new_column_pert_indices = _gen_empty_columns()

    for s in range(len(tokens_orig)):
        sentence = tokens_orig[s]
        sentence_pert = _custom_remove_char(sentence, PUNCTUATION)
        if sentence == sentence_pert:
            new_column_tokens.append(tokens_orig[s])
            new_column_concat.append(df.iloc[s, orig_index])
            new_column_success_flag.append(0)
            new_column_pert_indices.append(None)
        else:
            empty_indices = []
            # tracking the indices of tokens that have been removed
            for t in range(len(sentence)):
                if sentence_pert[t] == '':
                    empty_indices.append(t)
            new_column_tokens.append(sentence_pert)
            new_column_concat.append(" ".join(sentence_pert))
            new_column_success_flag.append(1)
            new_column_pert_indices.append(empty_indices)

    df['remove_all_punct_concat'] = new_column_concat
    df['remove_all_punct_tokens'] = new_column_tokens
    df['remove_all_punct_success'] = new_column_success_flag
    df['remove_all_punct_pert_ind'] = new_column_pert_indices


def switch_gender(df: pd.DataFrame, orig_index: int, tokens_orig: list):
    """
    Change gendered words

    :param df: DataFrame containing sentences
    :param orig_index: integer indicating the column index of the original sentence
    :param tokens_orig: tokenised version of sentence
    :return: None. Modifies DataFrame in-place
    """

    new_column_tokens, new_column_concat, new_column_success_flag, new_column_pert_indices = _gen_empty_columns()

    for s in range(len(tokens_orig)):
        sentence = deepcopy(tokens_orig[s])
        perturbed_indices = []
        for t in range(len(sentence)):
            if sentence[t] in DICT_GENDER:
                sentence[t] = DICT_GENDER[sentence[t]]
                perturbed_indices.append(t)
        new_column_tokens.append(sentence)
        if len(perturbed_indices) == 0:
            new_column_concat.append(df.iloc[s, orig_index])
            new_column_success_flag.append(0)
            new_column_pert_indices.append(None)
        else:
            new_column_concat.append(" ".join(sentence))
            new_column_success_flag.append(1)
            new_column_pert_indices.append(perturbed_indices)

    df['switch_gender_concat'] = new_column_concat
    df['switch_gender_tokens'] = new_column_tokens
    df['switch_gender_success'] = new_column_success_flag
    df['switch_gender_pert_ind'] = new_column_pert_indices


def add_perturbations(
        df: pd.DataFrame, tokenizer, sentence_col_name: str, perturbation_functions, seed=3
):
    """
    Apply multiple perturbations, generating a new column for each perturbation

    :param tokenizer: SpacyTokenizer or BertTokenizer
    :param df: DataFrame containing sentences
    :param sentence_col_name: Name of column containing sentence to be perturbed
    :param perturbation_functions: List of perturbation functions
    :param seed: Random seed
    :return: DataFrame with additional columns containing perturbed sentences and success flags
    """
    df = df.copy()

    tokenizer = tokenizer or SpacyTokenizer()

    orig_index = df.columns.get_loc(sentence_col_name)

    tokens_orig = [[str(x) for x in tokenizer.tokenize(df.iloc[i, orig_index])] for i in range(len(df))]

    df['tokens_orig'] = tokens_orig

    np.random.seed(seed)  # Set seed as some perturbations are stochastic

    for perturbation in perturbation_functions:
        perturbation(df, orig_index, tokens_orig)

    return df
