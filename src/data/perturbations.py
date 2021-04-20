import numpy as np
import pandas as pd
import spacy
import random
from checklist.perturb import Perturb
import urllib.request
import json 

from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer
from copy import deepcopy

nlp = spacy.load('en_core_web_sm', disable=["tagger", "ner", "lemmatizer"])

## reference objects to be used in perturbations

PUNCTUATION = ['!', '"', '&', "'", '(', ')', ',', '-', '.', '/', ':', ';', '?', '[', ']', '_', '`', '{', '}', '—',
 '…', '®', '–', '™', '‐']
"""list of characters to be removed in punctuation-related perturbations"""

DICT_GENDER = {
    'he': 'she', 
    'him':'her', 
    'his': 'her', 
    'she':'he', 
    'her': 'his', 
    'hers': 'his'}
pairs = [
    ['man','woman'],
    ['men','women'],
    ['boy','girl'],
    ['boyfriend','girlfriend'],
    ['wife', 'husband'], 
    ['brother','sister']]
for pair in pairs:
    DICT_GENDER[pair[0]] = pair[1]
    DICT_GENDER[pair[1]] = pair[0]
"""list of gendered words and pronouns to be replaced in related perturbations"""

data = urllib.request.urlopen('https://raw.githubusercontent.com/marcotcr/checklist/115f123de47ab015b2c3a6baebaffb40bab80c9f/checklist/data/names.json').read()
DICT_NAMES = json.loads(data)

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
    dummy_dict = dict.fromkeys(char,'')
    table = dummy_string.maketrans(dummy_dict)
    for i in range(len(text)):
        result.append(text[i].translate(table))
    if type(text_orig) == str:
        result = result[0]
    return result

def _gen_empty_columns():
    new_column_tokens = []
    new_column_concat = []
    new_column_success = []
    return new_column_tokens, new_column_concat, new_column_success

## perturbations

def checklist_contract_sentence(df, sentence_col_name):
    """
    Contract sentence length by using abbreviations e.g. "it is" to "it's"

    :param df: DataFrame containing sentences
    :param sentence_col_name: Name of column containing sentence to be perturbed
    :return: None. Modifies DataFrame in-place
    """
    sentences_contracted = [Perturb.contract(df[sentence_col_name][i]) for i in range(len(df))]

    df[sentence_col_name + '_contract_sent'] = sentences_contracted

    # Create success flag
    df['success_contract_sent'] = np.where(
        df[sentence_col_name + '_contract_sent'] != df[sentence_col_name],
        1,
        0
    )

def checklist_change_names(df, sentence_col_name):
    """
    Change name in sentence if one exists and if name is in CheckList's name lookup json

    :param df: DataFrame containing sentences
    :param sentence_col_name: Name of column containing sentence to be perturbed
    :return: None. Modifies DataFrame in-place
    """
    # Checklist requires pre-processing with Spacy for these perturbations
    pdata = list(nlp.pipe(df[sentence_col_name]))

    # Only returns sentences with successful perturbations else drops
    change_name_raw_output = Perturb.perturb(
        pdata,
        Perturb.change_names,
        keep_original=True,  # Keeps original sentence
        n=1  # Number of replacements to generate
    )['data']

    # key=original, value=perturbed
    original_to_pert = dict(
        zip(
            [sent_pair[0] for sent_pair in change_name_raw_output],
            [sent_pair[1] for sent_pair in change_name_raw_output]
        )
    )

    # Un-perturbed sentences will be null
    df[sentence_col_name + '_change_names_checklist'] = df[sentence_col_name].map(original_to_pert)

    # Success flag
    df['success_change_names_checklist'] = np.where(
        ~df[sentence_col_name + '_change_names_checklist'].isnull(),
        1,
        0
    )

    # Fill nulls with original
    df[sentence_col_name + '_change_names_checklist'] = np.where(
        df[sentence_col_name + '_change_names_checklist'].isnull(),
        df[sentence_col_name],
        df[sentence_col_name + '_change_names_checklist']
    )

def custom_change_names(df: pd.DataFrame, sentence_col_name: str, tokens_orig: list, dict_names: dict = DICT_NAMES):
    """
    Change name in sentence if one exists and if name is in CheckList's name lookup json

    :param df: DataFrame containing sentences
    :param sentence_col_name: Name of column containing sentence to be perturbed
    :param tokens_orig: tokenised version of sentence
    :return: None. Modifies DataFrame in-place
    """

    new_column_tokens, new_column_concat, new_column_success = _gen_empty_columns()

    name_categories = list(dict_names.keys())[:2]
 
    for s in range(len(tokens_orig)):
        sentence = tokens_orig[s]
        perturbed_indices = []
        for t in range(len(sentence)):
            for category in name_categories:
                if sentence[t] in dict_names[category]:
                    name = sentence[t]
                    while name == sentence[t]:
                        name_index = random.randint(0, len(dict_names[category])-1)
                        name = dict_names[category][name_index]
                    sentence[t] = name
                    perturbed_indices.append(t)
                    break
            # limit number of perturbations to 1
            if len(perturbed_indices) > 0:
                break
                    
        new_column_tokens.append(sentence)
        if len(perturbed_indices) == 0:
            new_column_concat.append(df[sentence_col_name][s])
            new_column_success.append(0)
        else:
            new_column_concat.append(" ".join(sentence))
            new_column_success.append(f'1, perturbed tokens: {perturbed_indices}')

    df[sentence_col_name + '_change_names_concat'] = new_column_concat
    df[sentence_col_name + '_change_names_tokens'] = new_column_tokens
    df['success_change_names'] = new_column_success

def custom_add_typo(df: pd.DataFrame, sentence_col_name: str, tokens_orig: list):
    """
    Add typo using checklist library

    :param df: DataFrame containing sentences
    :param sentence_col_name: Name of column containing sentence to be perturbed
    :param tokens_orig: tokenised version of sentence
    :return: None. Modifies DataFrame in-place
    """
    new_column_tokens, new_column_concat, new_column_success = _gen_empty_columns()

    for s in range(len(tokens_orig)):
        sentence = tokens_orig[s]
        token_length = 1
        attempts = 0
        while token_length <= 1 and attempts < 10:
            index_pert = random.randint(1, len(sentence) - 2)
            token_length = len(sentence[index_pert])
            attempts += 1
        if token_length <= 1:
            new_column_tokens.append(sentence)
            new_column_concat.append(df[sentence_col_name][s])
            new_column_success.append(0)
        else:
            sentence[index_pert+1] = sentence[index_pert][-1] + sentence[index_pert+1]
            sentence[index_pert] = sentence[index_pert][:-1]
            new_column_tokens.append(sentence)
            new_column_concat.append(" ".join(sentence))
            new_column_success.append(f'1, perturbed tokens: {index_pert, index_pert+1}')

    df[sentence_col_name + '_add_typo_concat'] = new_column_concat
    df[sentence_col_name + '_add_typo_tokens'] = new_column_tokens
    df['success_add_typo'] = new_column_success


def custom_strip_trailing_punct(df: pd.DataFrame, sentence_col_name: str, tokens_orig: list):
    """
    Remove punctuation at end of sentence

    :param df: DataFrame containing sentences
    :param sentence_col_name: Name of column containing sentence to be perturbed
    :param tokens_orig: tokenised version of sentence
    :return: None. Modifies DataFrame in-place
    """

    new_column_tokens, new_column_concat, new_column_success = _gen_empty_columns()

    for s in range(len(tokens_orig)):
        sentence = tokens_orig[s]
        if sentence[-1] in PUNCTUATION:
            sentence = sentence[:-1]
            new_column_concat.append(" ".join(sentence))
            new_column_success.append(1)
        else:
            new_column_concat.append(df[sentence_col_name][s])
            new_column_success.append(0)
        new_column_tokens.append(sentence)           

    df[sentence_col_name + '_strip_punct_concat'] = new_column_concat
    df[sentence_col_name + '_strip_punct_tokens'] = new_column_tokens
    df['success_strip_punct'] = new_column_success

def custom_remove_commas(df: pd.DataFrame, sentence_col_name: str, tokens_orig: list):
    """
    Remove all commas from sentence

    :param df: DataFrame containing sentences
    :param sentence_col_name: Name of column containing sentence to be perturbed
    :param tokens_orig: tokenised version of sentence
    :return: None. Modifies DataFrame in-place
    """

    new_column_tokens, new_column_concat, new_column_success = _gen_empty_columns()
    
    for s in range(len(tokens_orig)):
        sentence = tokens_orig[s]
        sentence_pert = _custom_remove_char(sentence, ',')
        if sentence == sentence_pert:
            new_column_tokens.append(tokens_orig[s])
            new_column_concat.append(df[sentence_col_name][s])
            new_column_success.append(0)
        else:
            empty_indices = []
            # tracking the indices of tokens that have been removed
            for t in range(len(sentence)):
                if sentence_pert[t] == '':
                    empty_indices.append(t)
            new_column_tokens.append(sentence_pert)
            new_column_concat.append(" ".join(sentence_pert))
            new_column_success.append(f'1, tokens removed: {empty_indices}')

    df[sentence_col_name + '_remove_commas_concat'] = new_column_concat
    df[sentence_col_name + '_remove_commas_tokens'] = new_column_tokens
    df['success_remove_commas'] = new_column_success

def custom_remove_all_punctuation(df: pd.DataFrame, sentence_col_name: str, tokens_orig: list):
    """
    Remove all punctuation from sentence

    :param df: DataFrame containing sentences
    :param sentence_col_name: Name of column containing sentence to be perturbed
    :param tokens_orig: tokenised version of sentence
    :return: None. Modifies DataFrame in-place
    """

    new_column_tokens, new_column_concat, new_column_success = _gen_empty_columns()
    
    for s in range(len(tokens_orig)):
        sentence = tokens_orig[s]
        sentence_pert = _custom_remove_char(sentence, PUNCTUATION)
        if sentence == sentence_pert:
            new_column_tokens.append(tokens_orig[s])
            new_column_concat.append(df[sentence_col_name][s])
            new_column_success.append(0)
        else:
            empty_indices = []
            # tracking the indices of tokens that have been removed
            for t in range(len(sentence)):
                if sentence_pert[t] == '':
                    empty_indices.append(t)
            new_column_tokens.append(sentence_pert)
            new_column_concat.append(" ".join(sentence_pert))
            new_column_success.append(f'1, tokens removed: {empty_indices}')
    
    df[sentence_col_name + '_remove_all_punct_concat'] = new_column_concat
    df[sentence_col_name + '_remove_all_punct_tokens'] = new_column_tokens
    df['success_remove_all_punct'] = new_column_success


def custom_switch_gender(df: pd.DataFrame, sentence_col_name: str, tokens_orig: list, dict_gender: dict = DICT_GENDER):
    """
    Change gendered words

    :param df: DataFrame containing sentences
    :param sentence_col_name: Name of column containing sentence to be perturbed
    :param tokens_orig: tokenised version of sentence
    :param dict_gender: look-up dict of words to change
    :return: None. Modifies DataFrame in-place
    """

    new_column_tokens, new_column_concat, new_column_success = _gen_empty_columns()

    for s in range(len(tokens_orig)):
        sentence = tokens_orig[s]
        perturbed_indices = []
        for t in range(len(sentence)):
            if sentence[t] in dict_gender:
                sentence[t] = dict_gender[sentence[t]]
                perturbed_indices.append(t)
        new_column_tokens.append(sentence)
        if len(perturbed_indices) == 0:
            new_column_concat.append(df[sentence_col_name][s])
            new_column_success.append(0)
        else:
            new_column_concat.append(" ".join(sentence))
            new_column_success.append(f'1, perturbed tokens: {perturbed_indices}')

    df[sentence_col_name + '_switch_gender_concat'] = new_column_concat
    df[sentence_col_name + '_switch_gender_tokens'] = new_column_tokens
    df['success_switch_gender'] = new_column_success

perturbations_with_tokenization = [custom_remove_commas, custom_remove_all_punctuation, custom_switch_gender, \
    custom_strip_trailing_punct, custom_add_typo, custom_change_names]
"""list of perturbations that require tokenisation"""

def add_perturbations(
        df: pd.DataFrame, sentence_col_name: str, perturbation_functions, seed=3, tokenizer = None
):
    """
    Apply multiple perturbations, generating a new column for each perturbation

    :param df: DataFrame containing sentences
    :param sentence_col_name: Name of column containing sentence to be perturbed
    :param perturbation_functions: List of perturbation functions
    :param seed: Random seed
    :return: DataFrame with additional columns containing perturbed sentences and success flags
    """
    df = df.copy()

    tokenizer = tokenizer or SpacyTokenizer()

    tokens_orig = [[str(x) for x in tokenizer.tokenize(df[sentence_col_name][i])] for i in range(len(df))]
    df[sentence_col_name + '_tokens'] = tokens_orig

    np.random.seed(seed)  # Set seed as some perturbations are stochastic

    for perturbation in perturbation_functions:
        if perturbation in perturbations_with_tokenization:
            perturbation(df, sentence_col_name, tokens_orig)
        else:
            perturbation(df, sentence_col_name)

    return df
