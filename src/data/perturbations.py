import numpy as np
import spacy
from checklist.perturb import Perturb

from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer
from copy import deepcopy

nlp = spacy.load('en_core_web_sm')

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

def custom_remove_char(text_orig, char):
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



## perturbations

def checklist_strip_punctuation(df, sentence_col_name):
    """
    Strip trailing punctuation

    :param df: DataFrame containing sentences
    :param sentence_col_name: Name of column containing sentence to be perturbed
    :return: None. Modifies DataFrame in-place
    """
    # Checklist requires pre-processing with Spacy for these perturbations
    pdata = list(nlp.pipe(df[sentence_col_name]))

    stripped_sentences = [Perturb.strip_punctuation(pdata[i]) for i in range(len(pdata))]

    df[sentence_col_name + '_strip_punct'] = stripped_sentences

    # Create success flag as not all sentences will contain punctuation:
    df['success_strip_punct'] = np.where(
        df[sentence_col_name + '_strip_punct'] != df[sentence_col_name],
        1,
        0
    )


def checklist_add_typos(df, sentence_col_name):
    """
    Adds typos by removing last letter of one word and appending to start of next

    :param df: DataFrame containing sentences
    :param sentence_col_name: Name of column containing sentence to be perturbed
    :return: None. Modifies DataFrame in-place
    """
    sentences_w_typos = [Perturb.add_typos(df[sentence_col_name][i]) for i in range(len(df))]

    df[sentence_col_name + '_add_typos'] = sentences_w_typos

    # Create success flag
    df['success_add_typos'] = np.where(
        df[sentence_col_name + '_add_typos'] != df[sentence_col_name],
        1,
        0
    )


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
    df[sentence_col_name + '_change_names'] = df[sentence_col_name].map(original_to_pert)

    # Success flag
    df['success_change_names'] = np.where(
        ~df[sentence_col_name + '_change_names'].isnull(),
        1,
        0
    )

    # Fill nulls with original
    df[sentence_col_name + '_change_names'] = np.where(
        df[sentence_col_name + '_change_names'].isnull(),
        df[sentence_col_name],
        df[sentence_col_name + '_change_names']
    )

def custom_remove_comma(df, sentence_col_name, tokens_orig):
    """
    Remove all commas from sentence

    :param df: DataFrame containing sentences
    :param sentence_col_name: Name of column containing sentence to be perturbed
    :param tokenizer to be applied to the sentence
    :return: None. Modifies DataFrame in-place
    """

    tokens_pert = [custom_remove_char(sentence, ',') for sentence in tokens_orig]

    empty_indices = []
    for i in range(len(tokens_pert)):
        if tokens_pert[i] == tokens_orig[i]:
            empty_indices.append(None)
        else:
            empty_indices_sentence = []
            for t in range(len(tokens_pert[i])):
                if tokens_pert[i][t] == '':
                    empty_indices_sentence.append(t)
            empty_indices.append(empty_indices_sentence)

    new_column_sentence = []
    new_column_success = []

    for i in range(len(tokens_pert)):
        if empty_indices[i] == None:
            new_column_sentence.append(df[sentence_col_name][i])
            new_column_success.append(0)
        
        else:
            new_column_sentence.append(" ".join(tokens_pert[i]))
            new_column_success.append(f'1, tokens removed: {empty_indices[i]}')
    
    df[sentence_col_name + '_remove_commas'] = new_column_sentence
    df['success_remove_commas'] = new_column_success

def custom_remove_all_punctuation(df, sentence_col_name, tokens_orig):
    """
    Remove all punctuation from sentence

    :param df: DataFrame containing sentences
    :param sentence_col_name: Name of column containing sentence to be perturbed
    :param tokenizer to be applied to the sentence
    :return: None. Modifies DataFrame in-place
    """

    tokens_pert = [custom_remove_char(sentence, PUNCTUATION) for sentence in tokens_orig]

    empty_indices = []
    for i in range(len(tokens_pert)):
        if tokens_pert[i] == tokens_orig[i]:
            empty_indices.append(None)
        else:
            empty_indices_sentence = []
            for t in range(len(tokens_pert[i])):
                if tokens_pert[i][t] == '':
                    empty_indices_sentence.append(t)
            empty_indices.append(empty_indices_sentence)

    new_column_sentence = []
    new_column_success = []

    for i in range(len(tokens_pert)):
        if empty_indices[i] == None:
            new_column_sentence.append(df[sentence_col_name][i])
            new_column_success.append(0)
        
        else:
            new_column_sentence.append(" ".join(tokens_pert[i]))
            new_column_success.append(f'1, tokens removed: {empty_indices[i]}')
    
    df[sentence_col_name + '_remove_all_punct'] = new_column_sentence
    df['success_remove_all_punct'] = new_column_success


def custom_switch_gender(df, sentence_col_name, tokens_orig, dict_gender = DICT_GENDER):
    """
    Change gendered words

    :param df: DataFrame containing sentences
    :param sentence_col_name: Name of column containing sentence to be perturbed
    :param dict_gender: look-up dict of words to change
    :param tokenizer to be applied to the sentence
    :return: None. Modifies DataFrame in-place
    """
    tokens = deepcopy(tokens_orig)

    new_column_sentence = [None for i in range(len(df))]
    new_column_success = [None for i in range(len(df))]

    for s in range(len(tokens)):
        sentence = tokens[s]
        changes = 0
        for i in range(len(sentence)):
            if sentence[i] in dict_gender:
                sentence[i] = dict_gender[sentence[i]]
                changes += 1
        if changes > 0:
            new_column_sentence[s] = " ".join(tokens[s])
            new_column_success[s] = 1
        else:
            new_column_sentence[s] = df[sentence_col_name][s]
            new_column_success[s] = 0

    df[sentence_col_name + '_switch_gender'] = new_column_sentence
    df['success_switch_gender'] = new_column_success

perturbations_with_tokenization = [custom_remove_comma, custom_remove_all_punctuation, custom_switch_gender]
"""list of perturbations that require tokenisation"""

def add_perturbations(
        df, sentence_col_name, perturbation_functions, seed=3, tokenizer = None
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

    # verify if any of the perturbation functions require tokenization
    if len(set(perturbation_functions + perturbations_with_tokenization)) != len(perturbation_functions) + len(perturbations_with_tokenization):
        tokens_orig = [[str(x) for x in tokenizer.tokenize(df[sentence_col_name][i])] for i in range(len(df))]

    np.random.seed(seed)  # Set seed as some perturbations are stochastic

    for perturbation in perturbation_functions:
        if perturbation in perturbations_with_tokenization:
            perturbation(df, sentence_col_name, tokens_orig)
        else:
            perturbation(df, sentence_col_name)

    return df
