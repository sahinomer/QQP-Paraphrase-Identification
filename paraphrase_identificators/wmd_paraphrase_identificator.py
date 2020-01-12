import numpy as np

import gensim.downloader as api
from gensim.models import KeyedVectors

from paraphrase_identificators.paraphrase_identificator import ParaphraseIdentificator
from qqp_dataframe import QQPDataFrame

import tensorflow.keras.backend as K
from keras import losses, metrics

from sklearn.metrics import log_loss, accuracy_score


class WordMoverDistanceParaphraseIdentificator(ParaphraseIdentificator):

    def initialize_dataset_frame(self, path, test_rate=0.1):
        self.qqp_df = QQPDataFrame(path=path)
        self.qqp_df.split_train_test(test_rate=test_rate)

    def initialize_model(self):
        self.model_name = 'word_mover_distance'
        path = api.load('word2vec-google-news-300', return_path=True)
        self.model = KeyedVectors.load_word2vec_format(path, binary=True)

    def test(self, batch_size=None):
        question1, question2, is_duplicate = self.qqp_df.get_raw_test_data()
        distance = np.zeros((len(is_duplicate),), dtype=float)
        for i, (q1, q2) in enumerate(zip(question1, question2)):
            distance[i] = self.model.wmdistance(q1, q2)

        distance = 1-np.minimum(distance, 1)
        loss = log_loss(is_duplicate, distance)
        acc = np.mean(np.equal(is_duplicate, np.round(distance)), axis=-1)

        return loss, acc

    def predict(self, question1, question2):
        return self.model.wmdistance(question1, question2)