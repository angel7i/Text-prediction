"""Module to load data.

Consists of functions to load data from four different datasets (IMDb, Rotten
Tomatoes, Tweet Weather, Amazon Reviews). Each of these functions do the
following:
    - Read the required fields (texts and labels).
    - Do any pre-processing if required. For example, make sure all label
        values are in range [0, num_classes-1].
    - Split the data into training and validation sets.
    - Shuffle the training data.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import pandas as pd


def load_tweet_weather_topic_classification_dataset(data_path,
                                                    validation_split=0.2,
                                                    seed=123):
    """Loads the tweet weather topic classification dataset.

    # Arguments
        data_path: string, path to the data directory.
        validation_split: float, percentage of data to use for validation.
        seed: int, seed for randomizer.

    # Returns
        A tuple of training and validation data.
        Number of training samples: 62356
        Number of test samples: 15590
        Number of topics: 15

    # References
        https://www.kaggle.com/c/crowdflower-weather-twitter/data

        Download from:
        https://www.kaggle.com/c/3586/download/train.csv
    """
    columns = [1] + [i for i in range(13, 28)]  # 1 - text, 13-28 - topics.
    data = _load_and_shuffle_data(data_path, 'train.csv', columns, seed)

    # Get tweet text and the max confidence score for the weather types.
    texts = list(data['tweet'])
    weather_data = data.iloc[:, 1:]

    labels = []
    for i in range(len(texts)):
        # Pick topic with the max confidence score.
        labels.append(np.argmax(list(weather_data.iloc[i, :].values)))

    return _split_training_and_validation_sets(
        texts, np.array(labels), validation_split)


def load_rotten_tomatoes_sentiment_analysis_dataset(data_path,
                                                    validation_split=0.2,
                                                    seed=123):
    """Loads the rotten tomatoes sentiment analysis dataset.

    # Arguments
        data_path: string, path to the data directory.
        validation_split: float, percentage of data to use for validation.
        seed: int, seed for randomizer.

    # Returns
        A tuple of training and validation data.
        Number of training samples: 124848
        Number of test samples: 31212
        Number of categories: 5 (0 - negative, 1 - somewhat negative,
                2 - neutral, 3 - somewhat positive, 4 - positive)

    # References
        https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data

        Download and uncompress archive from:
        https://www.kaggle.com/c/3810/download/train.tsv.zip
    """
    columns = (2, 3)  # 2 - Phrases, 3 - Sentiment.
    data = _load_and_shuffle_data(data_path, 'train.tsv', columns, seed, '\t')

    # Get the review phrase and sentiment values.
    texts = list(data['Phrase'])
    labels = np.array(data['Sentiment'])
    return _split_training_and_validation_sets(texts, labels, validation_split)


def _load_and_shuffle_data(data_path,
                           file_name,
                           cols,
                           seed,
                           separator=',',
                           header=0):
    """Loads and shuffles the dataset using pandas.

    # Arguments
        data_path: string, path to the data directory.
        file_name: string, name of the data file.
        cols: list, columns to load from the data file.
        seed: int, seed for randomizer.
        separator: string, separator to use for splitting data.
        header: int, row to use as data header.
    """
    np.random.seed(seed)
    data_path = os.path.join(data_path, file_name)
    data = pd.read_csv(data_path, usecols=cols, sep=separator, header=header)
    return data.reindex(np.random.permutation(data.index))


def _split_training_and_validation_sets(texts, labels, validation_split):
    """Splits the texts and labels into training and validation sets.

    # Arguments
        texts: list, text data.
        labels: list, label data.
        validation_split: float, percentage of data to use for validation.

    # Returns
        A tuple of training and validation data.
    """

    num_training_samples = int((1 - validation_split) * len(texts))
    return ((texts[:num_training_samples], labels[:num_training_samples]),
            (texts[num_training_samples:], labels[num_training_samples:]))
