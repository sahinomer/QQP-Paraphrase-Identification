import numpy as np

from embedding import WordEmbedding
from qqp_dataframe import QQPDataFrame

qqp_df = QQPDataFrame(path='train.csv')
qqp_df.split_train_test(test_rate=0.99)
qqp_df.fit_tokenizer()


word_embedding = WordEmbedding(embfile='../PassageQueryProject/glove.840B.300d.txt')
word_embedding.create_embedding_matrix(qqp_df.tokenizer)

q1, q2, d = qqp_df.get_train_data()
q1_seq_emb = word_embedding.sequences_to_embeddings(sequences=q1)

while True:
    # q1 = input('Q1:')
    # q2 = input('Q2:')

    questions = input('Q1+Q2:')

    q1 = questions.split('","')[0]
    q2 = questions.split('","')[1]
    print('Q1:', q1)
    print('Q2:', q2)

    q1, q2 = qqp_df.get_predict_data(q1, q2)

    is_duplicate = np.dot(q1, q2.T)
    print('is_duplicate:', is_duplicate)
    print('-' * 20)
