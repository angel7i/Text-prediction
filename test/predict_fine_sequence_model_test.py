import unittest

from tensorflow.python.keras.models import load_model
from keras_preprocessing.text import tokenizer_from_json
from tensorflow.python.keras.preprocessing import sequence

categories = ["clouds",
              "cold",
              "dry",
              "hot",
              "humid",
              "hurricane",
              "I can't tell",
              "ice",
              "other",
              "rain",
              "snow",
              "storms",
              "sun",
              "tornado",
              "wind"]
filename = '../models/tweet_weather_sepcnn_fine_tuned_model.h5'
vocabualry_file = '../models/tweet_weather_sepcnn_fine_tuned_vectorizer.json'

# Limit on the length of text sequences, obtained from training data.
MAX_LENGTH = 45


class ModelTest(unittest.TestCase):
    def setUp(self):
        self.model = load_model(filename)

        with open(vocabualry_file, "r") as json_file:
            json_content = json_file.read()

        self.tokenizer = tokenizer_from_json(json_content)

    def test_prediction(self):
        texts = [
            " I hope this weather clears up for the party tonight ",
        "At work looking out the window at the storm!!!# IM SCARED"]
        print("IN: ", texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        print("Text to Vector: ", sequences)
        print("Vector to Text: ", self.tokenizer.sequences_to_texts(sequences))
        fixed_sequences = sequence.pad_sequences(sequences, maxlen=MAX_LENGTH)
        print("Fixed vectors: ", fixed_sequences)
        predictions = self.model.predict(fixed_sequences)
        print("Predictions: ", predictions)

        for p in predictions:
            probabilities = list(p)
            index = probabilities.index(max(probabilities))
            print(categories[index])
