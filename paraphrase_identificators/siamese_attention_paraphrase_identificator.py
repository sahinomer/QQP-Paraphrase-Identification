import os
import pickle

from keras import Input, Model
from keras.activations import softmax
from keras.engine.saving import load_model
from keras.optimizers import Adam

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import BatchNormalization, Bidirectional, LSTM, dot, Dense, Dropout, multiply, concatenate, subtract, \
    Concatenate, Lambda, Dot, Permute, GlobalAvgPool1D, GlobalMaxPool1D, Flatten

from paraphrase_identificators.siamese_paraphrase_identificator import SiameseParaphraseIdentificator
from qqp_dataframe import QQPDataFrame


def unchanged_shape(input_shape):
    """Function for Lambda layer"""
    return input_shape


def soft_attention_alignment(input_1, input_2):
    """Align text representation with neural soft attention"""
    attention = Dot(axes=-1)([input_1, input_2])
    w_att_1 = Lambda(lambda x: softmax(x, axis=1),
                     output_shape=unchanged_shape)(attention)
    w_att_2 = Permute((2, 1))(Lambda(lambda x: softmax(x, axis=2),
                                     output_shape=unchanged_shape)(attention))
    in1_aligned = Dot(axes=1)([w_att_1, input_1])
    in2_aligned = Dot(axes=1)([w_att_2, input_2])
    return in1_aligned, in2_aligned


class SiameseAttentionParaphraseIdentificator(SiameseParaphraseIdentificator):

    def initialize_model(self):
        self.model_name = 'siamese_attention_dnn'

        lstm_network = Sequential(layers=[
            Embedding(self.word_embedding.vocabulary_size, self.word_embedding.dimensions,
                      weights=[self.word_embedding.embedding_matrix],
                      trainable=True, mask_zero=False),

            BatchNormalization(axis=2),

            Bidirectional(LSTM(256, return_sequences=True)),

        ])

        question1_input = Input(shape=(self.qqp_df.seq_len,), name='question1_input')
        question2_input = Input(shape=(self.qqp_df.seq_len,), name='question2_input')

        question1_lstm = lstm_network(question1_input)
        question2_lstm = lstm_network(question2_input)

        # Attention
        q1_aligned, q2_aligned = soft_attention_alignment(question1_lstm, question2_lstm)

        # Compose
        q1_sub = subtract([question1_lstm, q2_aligned])
        q1_mult = multiply([question1_lstm, q2_aligned])
        q1_submult = Concatenate()([q1_sub, q1_mult])

        q2_sub = subtract([question2_lstm, q1_aligned])
        q2_mult = multiply([question2_lstm, q1_aligned])
        q2_submult = Concatenate()([q2_sub, q2_mult])

        q1_combined = Concatenate()([question1_lstm, q2_aligned, q1_submult])
        q2_combined = Concatenate()([question2_lstm, q1_aligned, q2_submult])

        compose = Bidirectional(LSTM(256, return_sequences=True))
        q1_compare = compose(q1_combined)
        q2_compare = compose(q2_combined)

        # Aggregate
        q1_avg_pool = GlobalAvgPool1D()(q1_compare)
        q1_max_pool = GlobalMaxPool1D()(q1_compare)
        q1_rep = Concatenate()([q1_avg_pool, q1_max_pool])
        q2_avg_pool = GlobalAvgPool1D()(q2_compare)
        q2_max_pool = GlobalMaxPool1D()(q2_compare)
        q2_rep = Concatenate()([q2_avg_pool, q2_max_pool])

        # Classifier
        merged = Concatenate()([q1_rep, q2_rep])
        # merged = Flatten()(merged)

        dense = BatchNormalization()(merged)
        dense = Dense(128, activation='relu')(dense)
        dense = BatchNormalization()(dense)
        dense = Dropout(0.4)(dense)
        dense = Dense(128, activation='relu')(dense)
        dense = BatchNormalization()(dense)
        dense = Dropout(0.4)(dense)
        out_ = Dense(1, activation='sigmoid')(dense)

        self.model = Model(inputs=[question1_input, question2_input], outputs=out_)
        self.model.compile(optimizer=Adam(lr=1e-3),
                           loss='binary_crossentropy',
                           metrics=['binary_crossentropy', 'accuracy'])
