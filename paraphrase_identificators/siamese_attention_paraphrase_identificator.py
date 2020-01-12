from keras import Input, Model
from keras.optimizers import Adam

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import BatchNormalization, Bidirectional, LSTM, Dense, Dropout, multiply, subtract, \
    Concatenate, Lambda, Dot, Permute, GlobalAvgPool1D, GlobalMaxPool1D

import keras.backend as K

from paraphrase_identificators.siamese_paraphrase_identificator import SiameseParaphraseIdentificator


def unchanged_shape(input_shape):
    """Function for Lambda layer"""
    return input_shape


def soft_attention_alignment(input_1, input_2):

    """Align text representation with neural soft attention"""
    attention = Dot(axes=-1)([input_1, input_2])
    w_att_1 = Lambda(lambda x: K.softmax(x, axis=1),
                     output_shape=unchanged_shape)(attention)
    w_att_2 = Permute((2, 1))(Lambda(lambda x: K.softmax(x, axis=2),
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

            BatchNormalization(),

            Bidirectional(LSTM(256, return_sequences=True)),

        ])

        question1_input = Input(shape=(self.qqp_df.seq_len,), name='question1_input')
        question2_input = Input(shape=(self.qqp_df.seq_len,), name='question2_input')

        question1_lstm = lstm_network(question1_input)
        question2_lstm = lstm_network(question2_input)

        # Attention
        q1_aligned, q2_aligned = soft_attention_alignment(question1_lstm, question2_lstm)

        # Compose
        q1q2_sub = subtract([question1_lstm, q2_aligned])
        q1q2_mult = multiply([question1_lstm, q2_aligned])
        q1q2_submult = Concatenate()([q1q2_sub, q1q2_mult])

        q2q1_sub = subtract([question2_lstm, q1_aligned])
        q2q1_mult = multiply([question2_lstm, q1_aligned])
        q2q1_submult = Concatenate()([q2q1_sub, q2q1_mult])

        q1q2_combined = Concatenate()([question1_lstm, q2_aligned, q1q2_submult])
        q2q1_combined = Concatenate()([question2_lstm, q1_aligned, q2q1_submult])

        compose = Bidirectional(LSTM(256, return_sequences=True))
        q1q2_compare = compose(q1q2_combined)
        q2q1_compare = compose(q2q1_combined)

        # Aggregate
        q1q2_avg_pool = GlobalAvgPool1D()(q1q2_compare)
        q1q2_max_pool = GlobalMaxPool1D()(q1q2_compare)
        q1q2_concat = Concatenate()([q1q2_avg_pool, q1q2_max_pool])
        q2q1_avg_pool = GlobalAvgPool1D()(q2q1_compare)
        q2q1_max_pool = GlobalMaxPool1D()(q2q1_compare)
        q2q1_concat = Concatenate()([q2q1_avg_pool, q2q1_max_pool])

        # Classifier
        merged = Concatenate()([q1q2_concat, q2q1_concat])

        dense = BatchNormalization()(merged)
        dense = Dense(256, activation='relu')(dense)
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

