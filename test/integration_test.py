""" Module to test training accuracy.

The following tests exercise all modules by going through the complete ML
training process. We measure the accuracies at the end and check that they are
within +/- 2% of an expected number.
"""
import load_data
import train_fine_tuned_sequence_model
import train_sequence_model
import unittest

data_dir = '../data/'
embedding_data_dir = '../data/'


class TrainingTest(unittest.TestCase):

    def test_train_sequence_model(self):
        data = load_data.load_rotten_tomatoes_sentiment_analysis_dataset(data_dir)
        acc, loss = train_sequence_model.train_sequence_model(data)
        self.assertTrue(0.66 < acc < 0.70)
        self.assertTrue(0.80 < loss < 0.84)

    def test_train_fine_tuned_sequence_model(self):
        data = load_data.load_tweet_weather_topic_classification_dataset(data_dir)
        acc, loss = train_fine_tuned_sequence_model.train_fine_tuned_sequence_model(
            data, embedding_data_dir)
        self.assertTrue(0.82 < acc < 0.86)
        self.assertTrue(0.53 < loss < 0.57)
