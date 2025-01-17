
import os
import pickle

from keras import Input, Model
from keras.engine.saving import load_model
from keras.optimizers import Adam

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import BatchNormalization, Bidirectional, LSTM, Lambda, Dense, Dropout, \
    dot, multiply, concatenate, subtract

from paraphrase_identificators.paraphrase_identificator import ParaphraseIdentificator
from qqp_dataframe import QQPDataFrame
from utils import euclidean_similarity, euclidean_output_shape, contrastive_loss


class SiameseParaphraseIdentificator(ParaphraseIdentificator):

    def initialize_model(self):
        self.model_name = 'siamese_dnn'

        lstm_network = Sequential(layers=[
            Embedding(self.word_embedding.vocabulary_size, self.word_embedding.dimensions,
                      weights=[self.word_embedding.embedding_matrix],
                      trainable=True, mask_zero=False),

            BatchNormalization(),

            Bidirectional(LSTM(256, return_sequences=False)),

        ])

        question1_input = Input(shape=(self.qqp_df.seq_len,), name='question1_input')
        question2_input = Input(shape=(self.qqp_df.seq_len,), name='question2_input')

        question1_lstm = lstm_network(question1_input)
        question2_lstm = lstm_network(question2_input)

        # similarity = Lambda(euclidean_similarity,
        #                     output_shape=euclidean_output_shape)([question1_lstm, question2_lstm])

        dot_questions = dot([question1_lstm, question2_lstm], axes=1, normalize=True)
        subtract_q1q2 = subtract([question1_lstm, question2_lstm])
        subtract_q2q1 = subtract([question2_lstm, question1_lstm])
        multiply_questions = multiply([question1_lstm, question2_lstm])

        merged = concatenate([dot_questions, subtract_q1q2, subtract_q2q1, multiply_questions])

        dense = BatchNormalization()(merged)
        dense = Dense(128, activation='relu')(dense)
        dense = BatchNormalization()(dense)
        dense = Dropout(0.4)(dense)
        dense = Dense(128, activation='relu')(dense)
        dense = BatchNormalization()(dense)
        dense = Dropout(0.4)(dense)
        out = Dense(1, activation='sigmoid')(dense)

        self.model = Model(inputs=[question1_input, question2_input], outputs=out)
        self.model.compile(optimizer=Adam(lr=1e-3),
                           loss='binary_crossentropy',
                           metrics=['binary_crossentropy', 'binary_accuracy'])

    def train(self, epochs=10, batch_size=64):
        question1, question2, is_duplicate = self.qqp_df.get_train_data()

        self.model.fit(x=[question1, question2], y=is_duplicate, epochs=epochs, batch_size=batch_size,
                       validation_split=0.1, verbose=1,
                       class_weight=self.qqp_df.class_weights)

    def test(self, batch_size=64):
        question1, question2, is_duplicate = self.qqp_df.get_test_data()

        return self.model.evaluate(x=[question1, question2], y=is_duplicate, batch_size=batch_size, verbose=2)

    def predict(self, question1, question2):
        question1, question2 = self.qqp_df.get_prediction_data(question1, question2)
        return self.model.predict(x=[question1, question2])

    def save(self, path):

        # Create directory if not exist
        if not os.path.exists(path):
            os.makedirs(path)

        # Serialize tokenizer
        with open(path + '/tokenizer.pickle', 'wb') as handle:
            pickle.dump(self.qqp_df.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.model.save(path + '/weights.h5')

    def load(self, path):
        # Load tokenizer
        with open(path + '/tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)

        if self.qqp_df is None:
            self.qqp_df = QQPDataFrame()

        self.qqp_df.tokenizer = tokenizer
        self.model = load_model(path + '/weights.h5')
