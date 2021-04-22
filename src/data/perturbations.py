import numpy as np
import pandas as pd
import spacy
import random
from copy import deepcopy

from checklist.perturb import Perturb
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer
from nltk.corpus import wordnet as wn

nlp = spacy.load('en_core_web_sm')

## reference objects to be used in perturbations

PUNCTUATION = ['!', '"', '&', "'", '(', ')', ',', '-', '.', '/', ':', ';', '?', '[', ']', '_', '`', '{', '}', '—',
               '…', '®', '–', '™', '‐']
"""list of characters to be removed in punctuation-related perturbations"""

DICT_GENDER = {
    'he': 'she',
    'him': 'her',
    'his': 'her',
    'she': 'he',
    'her': 'his',
    'hers': 'his'}
pairs = [
    ['man', 'woman'],
    ['men', 'women'],
    ['boy', 'girl'],
    ['boyfriend', 'girlfriend'],
    ['wife', 'husband'],
    ['brother', 'sister']]
for pair in pairs:
    DICT_GENDER[pair[0]] = pair[1]
    DICT_GENDER[pair[1]] = pair[0]
"""list of gendered words and pronouns to be replaced in related perturbations"""

# dict taken from checklist source code
CONTRACTION_MAP = {
    'is not': "isn't", 'are not': "aren't", 'cannot': "can't",
    'could not': "couldn't", 'did not': "didn't", 'does not':
        "doesn't", 'do not': "don't", 'had not': "hadn't", 'has not':
        "hasn't", 'have not': "haven't", 'he is': "he's", 'how did':
        "how'd", 'how is': "how's", 'I would': "I'd", 'I will': "I'll",
    'I am': "I'm", 'i would': "i'd", 'i will': "i'll", 'i am': "i'm",
    'it would': "it'd", 'it will': "it'll", 'it is': "it's",
    'might not': "mightn't", 'must not': "mustn't", 'need not': "needn't",
    'ought not': "oughtn't", 'shall not': "shan't", 'she would': "she'd",
    'she will': "she'll", 'she is': "she's", 'should not': "shouldn't",
    'that would': "that'd", 'that is': "that's", 'there would':
        "there'd", 'there is': "there's", 'they would': "they'd",
    'they will': "they'll", 'they are': "they're", 'was not': "wasn't",
    'we would': "we'd", 'we will': "we'll", 'we are': "we're", 'were not':
        "weren't", 'what are': "what're", 'what is': "what's", 'when is':
        "when's", 'where did': "where'd", 'where is': "where's",
    'who will': "who'll", 'who is': "who's", 'who have': "who've", 'why is':
        "why's", 'will not': "won't", 'would not': "wouldn't", 'you would':
        "you'd", 'you will': "you'll", 'you are': "you're",
}

# update the format of the dict to split values into separate tokens
CONTRACTION_MAP = {key: [value.partition("'")[0], "".join(value.partition("'")[1:])] for key, value in
                   CONTRACTION_MAP.items()}

REVERSE_CONTRACTION_MAP = {
    "ain't": "is not", "aren't": "are not", "can't": "cannot",
    "can't've": "cannot have", "could've": "could have", "couldn't":
        "could not", "didn't": "did not", "doesn't": "does not", "don't":
        "do not", "hadn't": "had not", "hasn't": "has not", "haven't":
        "have not", "he'd": "he would", "he'd've": "he would have",
    "he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y":
        "how do you", "how'll": "how will", "how's": "how is",
    "I'd": "I would", "I'll": "I will", "I'm": "I am",
    "I've": "I have", "i'd": "i would", "i'll": "i will",
    "i'm": "i am", "i've": "i have", "isn't": "is not",
    "it'd": "it would", "it'll": "it will", "it's": "it is", "ma'am":
        "madam", "might've": "might have", "mightn't": "might not",
    "must've": "must have", "mustn't": "must not", "needn't":
        "need not", "oughtn't": "ought not", "shan't": "shall not",
    "she'd": "she would", "she'll": "she will", "she's": "she is",
    "should've": "should have", "shouldn't": "should not", "that'd":
        "that would", "that's": "that is", "there'd": "there would",
    "there's": "there is", "they'd": "they would",
    "they'll": "they will", "they're": "they are",
    "they've": "they have", "wasn't": "was not", "we'd": "we would",
    "we'll": "we will", "we're": "we are", "we've": "we have",
    "weren't": "were not", "what're": "what are", "what's": "what is",
    "when's": "when is", "where'd": "where did", "where's": "where is",
    "where've": "where have", "who'll": "who will", "who's": "who is",
    "who've": "who have", "why's": "why is", "won't": "will not",
    "would've": "would have", "wouldn't": "would not",
    "you'd": "you would", "you'd've": "you would have",
    "you'll": "you will", "you're": "you are", "you've": "you have"
}
# update the format of the dict to split values into separate tokens
REVERSE_CONTRACTION_MAP = {key: value.split() for key, value in REVERSE_CONTRACTION_MAP.items()}

ADJECTIVES = []
for synset in wn.all_synsets('a'):
    ADJECTIVES.extend(synset.lemma_names())
# remove duplicatives
ADJECTIVES = list(set(ADJECTIVES))
ADJECTIVES.remove('even')


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
    new_column_success = []
    return new_column_tokens, new_column_concat, new_column_success


def swap_adjectives(df: pd.DataFrame, sentence_col_name: str, tokens_orig: list):
    """
    swap two consecutive adjectives (connected by 'and' or 'or')

    :param df: DataFrame containing sentences
    :param sentence_col_name: Name of column containing sentence to be perturbed
    :param tokens_orig: tokenised version of sentence
    :return: None. Modifies DataFrame in-place
    """

    new_column_tokens, new_column_concat, new_column_success = _gen_empty_columns()

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
            new_column_concat.append(df[sentence_col_name][s])
            new_column_success.append(0)
        else:
            new_column_concat.append(" ".join(sentence))
            new_column_success.append([1, [pert_indices]])

    df[sentence_col_name + '_swap_adj_concat'] = new_column_concat
    df[sentence_col_name + '_swap_adj_tokens'] = new_column_tokens
    df['success_swap_adj'] = new_column_success


def contraction(df: pd.DataFrame, sentence_col_name: str, tokens_orig: list):
    """
    Change to and from contracted form (e.g. "you're" to "you are", or "you are" to "you're")

    :param df: DataFrame containing sentences
    :param sentence_col_name: Name of column containing sentence to be perturbed
    :param tokens_orig: tokenised version of sentence
    :return: None. Modifies DataFrame in-place
    """

    new_column_tokens, new_column_concat, new_column_success = _gen_empty_columns()

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
            new_column_concat.append(df[sentence_col_name][s])
            new_column_success.append(0)
        else:
            new_column_tokens.append(sentence)
            new_column_concat.append(" ".join(sentence))
            new_column_success.append([1, [pert_indices]])

    df[sentence_col_name + '_contraction_concat'] = new_column_concat
    df[sentence_col_name + '_contraction_tokens'] = new_column_tokens
    df['success_contraction'] = new_column_success


def change_first_name(df: pd.DataFrame, sentence_col_name: str, tokens_orig: list):
    """
    Change first name in sentence if one exists and if name is in CheckList's name lookup json

    :param df: DataFrame containing sentences
    :param sentence_col_name: Name of column containing sentence to be perturbed
    :param tokens_orig: tokenised version of sentence
    :return: None. Modifies DataFrame in-place
    """

    new_column_tokens, new_column_concat, new_column_success = _gen_empty_columns()
    # Checklist requires pre-processing with Spacy for this perturbation
    pdata = list(nlp.pipe(df[sentence_col_name]))

    for s in range(len(tokens_orig)):
        sentence = tokens_orig[s]
        new_sentence = Perturb.change_names(pdata[s], n=1, first_only=True, meta=True)
        if not new_sentence:
            new_column_tokens.append(sentence)
            new_column_concat.append(df[sentence_col_name][s])
            new_column_success.append(0)
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
                new_column_success.append([1, [token_index]])
            # if token cannot be found in original list of tokens
            else:
                new_column_tokens.append(sentence)
                new_column_concat.append(df[sentence_col_name][s])
                new_column_success.append(0)

    df[sentence_col_name + '_change_first_name_concat'] = new_column_concat
    df[sentence_col_name + '_change_first_name_tokens'] = new_column_tokens
    df['success_change_first_name'] = new_column_success


def change_last_name(df: pd.DataFrame, sentence_col_name: str, tokens_orig: list):
    """
    Change last name in sentence if one exists and if name is in CheckList's name lookup json

    :param df: DataFrame containing sentences
    :param sentence_col_name: Name of column containing sentence to be perturbed
    :param tokens_orig: tokenised version of sentence
    :return: None. Modifies DataFrame in-place
    """

    new_column_tokens, new_column_concat, new_column_success = _gen_empty_columns()
    # Checklist requires pre-processing with Spacy for this perturbation
    pdata = list(nlp.pipe(df[sentence_col_name]))

    for s in range(len(tokens_orig)):
        sentence = tokens_orig[s]
        new_sentence = Perturb.change_names(pdata[s], n=1, last_only=True, meta=True)
        if not new_sentence:
            new_column_tokens.append(sentence)
            new_column_concat.append(df[sentence_col_name][s])
            new_column_success.append(0)
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
                new_column_success.append([1, [token_index]])
            else:
                new_column_tokens.append(sentence)
                new_column_concat.append(df[sentence_col_name][s])
                new_column_success.append(0)

    df[sentence_col_name + '_change_last_name_concat'] = new_column_concat
    df[sentence_col_name + '_change_last_name_tokens'] = new_column_tokens
    df['success_change_last_name'] = new_column_success


def change_location(df: pd.DataFrame, sentence_col_name: str, tokens_orig: list):
    """
    Change location in sentence if one exists and if location is in CheckList's location lookup json

    :param df: DataFrame containing sentences
    :param sentence_col_name: Name of column containing sentence to be perturbed
    :param tokens_orig: tokenised version of sentence
    :return: None. Modifies DataFrame in-place
    """

    new_column_tokens, new_column_concat, new_column_success = _gen_empty_columns()
    # Checklist requires pre-processing with Spacy for this perturbation
    pdata = list(nlp.pipe(df[sentence_col_name]))

    for s in range(len(tokens_orig)):
        sentence = tokens_orig[s]
        new_sentence = Perturb.change_location(pdata[s], n=1, meta=True)
        if not new_sentence:
            new_column_tokens.append(sentence)
            new_column_concat.append(df[sentence_col_name][s])
            new_column_success.append(0)
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
                new_column_success.append([1, [token_index]])
            # if token cannot be found in original list of tokens
            else:
                new_column_tokens.append(sentence)
                new_column_concat.append(df[sentence_col_name][s])
                new_column_success.append(0)

    df[sentence_col_name + '_change_location_concat'] = new_column_concat
    df[sentence_col_name + '_change_location_tokens'] = new_column_tokens
    df['success_change_location_name'] = new_column_success


def add_typo(df: pd.DataFrame, sentence_col_name: str, tokens_orig: list):
    """
    Add typo using checklist library

    :param df: DataFrame containing sentences
    :param sentence_col_name: Name of column containing sentence to be perturbed
    :param tokens_orig: tokenised version of sentence
    :return: None. Modifies DataFrame in-place
    """
    new_column_tokens, new_column_concat, new_column_success = _gen_empty_columns()

    for s in range(len(tokens_orig)):
        sentence = deepcopy(tokens_orig[s])
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
            sentence[index_pert + 1] = sentence[index_pert][-1] + sentence[index_pert + 1]
            sentence[index_pert] = sentence[index_pert][:-1]
            new_column_tokens.append(sentence)
            new_column_concat.append(" ".join(sentence))
            new_column_success.append([1, [index_pert, index_pert + 1]])

    df[sentence_col_name + '_add_typo_concat'] = new_column_concat
    df[sentence_col_name + '_add_typo_tokens'] = new_column_tokens
    df['success_add_typo'] = new_column_success


def strip_trailing_punct(df: pd.DataFrame, sentence_col_name: str, tokens_orig: list):
    """
    Remove punctuation at end of sentence

    :param df: DataFrame containing sentences
    :param sentence_col_name: Name of column containing sentence to be perturbed
    :param tokens_orig: tokenised version of sentence
    :return: None. Modifies DataFrame in-place
    """

    new_column_tokens, new_column_concat, new_column_success = _gen_empty_columns()

    for s in range(len(tokens_orig)):
        sentence = deepcopy(tokens_orig[s])
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


def remove_commas(df: pd.DataFrame, sentence_col_name: str, tokens_orig: list):
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
            new_column_success.append([1, [empty_indices]])

    df[sentence_col_name + '_remove_commas_concat'] = new_column_concat
    df[sentence_col_name + '_remove_commas_tokens'] = new_column_tokens
    df['success_remove_commas'] = new_column_success


def remove_all_punctuation(df: pd.DataFrame, sentence_col_name: str, tokens_orig: list):
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
            new_column_success.append([1, [empty_indices]])

    df[sentence_col_name + '_remove_all_punct_concat'] = new_column_concat
    df[sentence_col_name + '_remove_all_punct_tokens'] = new_column_tokens
    df['success_remove_all_punct'] = new_column_success


def switch_gender(df: pd.DataFrame, sentence_col_name: str, tokens_orig: list, dict_gender: dict = DICT_GENDER):
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
        sentence = deepcopy(tokens_orig[s])
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
            new_column_success.append([1, [perturbed_indices]])

    df[sentence_col_name + '_switch_gender_concat'] = new_column_concat
    df[sentence_col_name + '_switch_gender_tokens'] = new_column_tokens
    df['success_switch_gender'] = new_column_success


def add_perturbations(
        df: pd.DataFrame, sentence_col_name: str, perturbation_functions, seed=3, tokenizer=None
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

    sentence_array = df[sentence_col_name].values

    tokens_orig = [tokenizer.tokenize(sentence) for sentence in sentence_array]
    tokens_orig = [[str(token) for token in raw_token_list] for raw_token_list in tokens_orig]
    df[sentence_col_name + '_tokens'] = tokens_orig

    np.random.seed(seed)  # Set seed as some perturbations are stochastic

    for perturbation in perturbation_functions:
        perturbation(df, sentence_col_name, tokens_orig)

    return df
