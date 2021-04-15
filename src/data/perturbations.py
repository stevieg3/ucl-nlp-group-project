import numpy as np
import spacy
from checklist.perturb import Perturb

nlp = spacy.load('en_core_web_sm')


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


def add_checklist_perturbations(
        df, sentence_col_name, perturbation_functions, seed=3
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

    np.random.seed(seed)  # Set seed as some perturbations are stochastic

    for perturbation in perturbation_functions:
        perturbation(df, sentence_col_name)

    return df
