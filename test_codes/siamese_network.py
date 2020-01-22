import os
import pickle

import numpy as np
import pandas as pd

from keras import Input, Model, losses, metrics, optimizers
from keras.activations import sigmoid, softmax
from keras.engine.saving import load_model
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import BatchNormalization, Bidirectional, LSTM, CuDNNLSTM, dot, Dense, Dropout, Flatten, Dot, Lambda, \
    Permute, Concatenate, Multiply, Add, GlobalAvgPool1D, GlobalMaxPool1D, multiply, concatenate, subtract

from embedding import WordEmbedding


import tensorflow as tf

from qqp_dataframe import QQPDataFrame

tf.keras.backend.clear_session()  # For easy reset of notebook state.


# def unchanged_shape(input_shape):
#     "Function for Lambda layer"
#     return input_shape
#
#
# def soft_attention_alignment(input_1, input_2):
#     "Align text representation with neural soft attention"
#     attention = Dot(axes=-1)([input_1, input_2])
#     w_att_1 = Lambda(lambda x: softmax(x, axis=1),
#                      output_shape=unchanged_shape)(attention)
#     w_att_2 = Permute((2,1))(Lambda(lambda x: softmax(x, axis=2),
#                              output_shape=unchanged_shape)(attention))
#     in1_aligned = Dot(axes=1)([w_att_1, input_1])
#     in2_aligned = Dot(axes=1)([w_att_2, input_2])
#     return in1_aligned, in2_aligned
#
#
# def substract(input_1, input_2):
#     "Substract element-wise"
#     neg_input_2 = Lambda(lambda x: -x, output_shape=unchanged_shape)(input_2)
#     out_ = Add()([input_1, neg_input_2])
#     return out_
#
#
# def submult(input_1, input_2):
#     "Get multiplication and subtraction then concatenate results"
#     mult = Multiply()([input_1, input_2])
#     sub = substract(input_1, input_2)
#     out_= Concatenate()([sub, mult])
#     return out_
#
#
# def apply_multiple(input_, layers):
#     "Apply layers to input then concatenate result"
#     if not len(layers) > 1:
#         raise ValueError('Layers list should contain more than 1 layer')
#     else:
#         agg_ = []
#         for layer in layers:
#             agg_.append(layer(input_))
#         out_ = Concatenate()(agg_)
#     return out_


def create_network(word_embedding, input_length):
    # lstm_network = create_lstm_base(word_embedding)

    lstm_network = Sequential(layers=[
        Embedding(word_embedding.vocabulary_size, word_embedding.dimensions,
                  weights=[word_embedding.embedding_matrix],
                  trainable=True, mask_zero=False),

        BatchNormalization(axis=2),

        Bidirectional(LSTM(256, return_sequences=False)),

    ])

    question1_input = Input(shape=(input_length,), name='question1_input')
    question2_input = Input(shape=(input_length,), name='question2_input')

    question1_lstm = lstm_network(question1_input)
    question2_lstm = lstm_network(question2_input)

    substract_questions = subtract([question1_lstm, question2_lstm])
    multiply_questions = multiply([question1_lstm, question2_lstm])
    dot_questinons = dot([question1_lstm, question2_lstm], axes=1, normalize=True)

    merged = concatenate([substract_questions, multiply_questions, dot_questinons])

    # # Attention
    # q1_aligned, q2_aligned = soft_attention_alignment(question1_lstm, question2_lstm)
    #
    # # Compose
    # q1_combined = Concatenate()([question1_lstm, q2_aligned, submult(question1_lstm, q2_aligned)])
    # q2_combined = Concatenate()([question2_lstm, q1_aligned, submult(question2_lstm, q1_aligned)])
    #
    # compose = Bidirectional(LSTM(256, return_sequences=True))
    # q1_compare = compose(q1_combined)
    # q2_compare = compose(q2_combined)
    #
    # # Aggregate
    # q1_rep = apply_multiple(q1_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])
    # q2_rep = apply_multiple(q2_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])
    #
    # # Classifier
    # merged = Concatenate()([q1_rep, q2_rep])

    dense = BatchNormalization()(merged)
    dense = Dense(128, activation='relu')(dense)
    dense = BatchNormalization()(dense)
    dense = Dropout(0.4)(dense)
    dense = Dense(128, activation='relu')(dense)
    dense = BatchNormalization()(dense)
    dense = Dropout(0.4)(dense)
    out_ = Dense(1, activation='sigmoid')(dense)

    model = Model(inputs=[question1_input, question2_input], outputs=out_)
    model.compile(optimizer=Adam(lr=1e-3), loss='binary_crossentropy', metrics=['binary_crossentropy', 'accuracy'])
    return model


def model_save(path, tokenizer, model):
    # Create directory if not exist
    if not os.path.exists(path):
        os.makedirs(path)

    # Serialize tokenizer
    with open(path + '/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    model.save(path + '/weights.h5')


def model_load(path):

    # Load tokenizer
    with open(path + '/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # returns a compiled model
    model = load_model(path + '/weights.h5')

    return tokenizer, model


if __name__ == "__main__":
    train = True
    if train:
        # qqp_df = pd.read_csv("train.csv").fillna("")
        # qqp_df = qqp_df[:100000]
        #
        # tokenizer = Tokenizer(lower=True)
        # tokenizer.fit_on_texts(qqp_df['question1'])
        # tokenizer.fit_on_texts(qqp_df['question2'])
        #
        # question1 = tokenizer.texts_to_sequences(qqp_df['question1'])
        # question1 = pad_sequences(question1, maxlen=50)
        #
        # question2 = tokenizer.texts_to_sequences(qqp_df['question2'])
        # question2 = pad_sequences(question2, maxlen=50)
        #
        # is_duplicate = np.asarray(qqp_df['is_duplicate'])
        #
        # print('Duplicated Rate:', is_duplicate.sum(), '/', len(is_duplicate), '=', is_duplicate.sum() / len(is_duplicate))

        qqp_df = QQPDataFrame(path='../train.csv')
        qqp_df.split_train_test(test_rate=0.95)
        qqp_df.fit_tokenizer()
        question1, question2, is_duplicate = qqp_df.get_train_data()

        word_embedding = WordEmbedding(embfile='../PassageQueryProject/glove.840B.300d.txt')
        word_embedding.create_embedding_matrix(qqp_df.tokenizer)

        model = create_network(word_embedding=word_embedding, input_length=50)

        # model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        model.compile(loss=[losses.binary_crossentropy],
                      optimizer='adam',
                      metrics=[metrics.binary_accuracy])
        model.summary()

        model.fit(x=[question1, question2], y=is_duplicate, batch_size=64, epochs=1,
                  validation_split=0.1, verbose=1)

        model_save('model', qqp_df.tokenizer, model)

    else:
        tokenizer, model = model_load(path='model')
        while True:
            # q1 = input('Q1:')
            # q2 = input('Q2:')

            questions = input('Q1+Q2:')

            q1 = questions.split('","')[0]
            q2 = questions.split('","')[1]
            print('Q1:', q1)
            print('Q2:', q2)

            q1_seq = tokenizer.texts_to_sequences([q1])
            q2_seq = tokenizer.texts_to_sequences([q2])

            q1_seq = pad_sequences(q1_seq, maxlen=50)
            q2_seq = pad_sequences(q2_seq, maxlen=50)

            is_duplicate = model.predict([q1_seq, q2_seq])
            print('is_duplicate:', is_duplicate)
            print('-'*20)



