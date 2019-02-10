import unittest

from tensorflow.python.keras.models import load_model
from keras_preprocessing.text import tokenizer_from_json
from tensorflow.python.keras.preprocessing import sequence

categories = ["negative",
              "somewhat negative",
              "neutral",
              "somewhat positive",
              "positive"]
filename = "../models/rotten_tomatoes_sepcnn_model.h5"
vocabualry_file = "../models/rotten_tomatoes_sepcnn_vectorizer.json"

# Limit on the length of text sequences, obtained from training data.
MAX_LENGTH = 49


class ModelTest(unittest.TestCase):
    def setUp(self):
        self.model = load_model(filename)

        with open(vocabualry_file, "r") as json_file:
            json_content = json_file.read()

        self.tokenizer = tokenizer_from_json(json_content)

    def test_prediction(self):
        texts = [
            "A comedy-drama of nearly epic proportions rooted in a sincere performance by the title character undergoing midlife crisis",
            "Aggressive self-glorification and a manipulative whitewash",
            " that movie was more of the same, the final was awful"]
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
