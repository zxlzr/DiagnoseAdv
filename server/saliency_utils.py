import numpy as np


def normalize(raw_saliency):
    # Normalize the raw saliency scores by squared sum and min-max normalization
    raw_saliency = np.sqrt((raw_saliency ** 2).sum(axis=1))
    return (raw_saliency - raw_saliency.min()) / (raw_saliency.max() - raw_saliency.min())


def merge_token_saliency(token_saliency):
    # Merge tokens to words and saliency for words
    word_saliency = []
    curr_word, curr_saliency = '', 0.0
    for tok, sal in token_saliency:
        if tok.startswith('##'):
            curr_word += tok[2:]
            curr_saliency = max(curr_saliency, sal)
        else:
            if curr_word:
                word_saliency.append((curr_word, curr_saliency))
            curr_word = tok
            curr_saliency = sal
    word_saliency.append((curr_word, curr_saliency))
    return word_saliency
