"""
Utility functions for optics trainers
"""
from collections import Counter
import numpy as np
import pandas as pd
import glob
# from external_mapping import instance_video_map

def merge_dicts(dicts):
    """
    The merge_dicts function takes a list of dictionaries as input and returns a single dictionary.
    The function combines the values for each key in the dictionaries, resulting in one final dictionary.

    Args:
        dicts: Pass a list of dictionaries to the function

    Returns:
        A dictionary

    Doc Author:
        Trelent
    """
    c = Counter()
    for d in dicts:
        c.update(d)
    return dict(c)

def jaccard_score_custom(list1, list2):
    """
    The jaccard_score_custom function takes two lists as input and returns the jaccard score between them.
    The jaccard score is defined as the intersection of two sets divided by their union.
    If either list is empty, then the function will return 0.

    Args:
        list1: Represent the list of words in a document
        list2: Compare the list of predicted labels with

    Returns:
        The jaccard score between two lists

    Doc Author:
        Trelent
    """

    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    if union==0:
        return 0.
    return float(intersection) / union

def aggregate_ts_scores(df_otc_out):
    """
    The aggregate_ts_scores function takes in a dataframe of audio features and returns a dataframe with the
    audio features aggregated by percentile. The function also sorts the values in descending order, so that
    the most important feature is listed first.

    Args:
        df_otc_out: Store the audio features for each video

    Returns:
        A dataframe with the mean of each row, sorted by descending values

    Doc Author:
        Trelent
    """
    ptile_10 = int(df_otc_out.shape[1] / 10)
    if ptile_10 > 0:
        df_otc_out = df_otc_out.iloc[:, ptile_10:-ptile_10]
    else:
        df_otc_out = df_otc_out.copy()
    df_out = pd.DataFrame(np.percentile(df_otc_out.values,50,axis=1), index=df_otc_out.index, columns=[1]).sort_values([1],ascending=False)
    df_out[1] = df_out[1]/df_out[1].sum()
    return df_out

