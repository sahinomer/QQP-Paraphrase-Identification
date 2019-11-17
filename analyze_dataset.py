import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer


qqp_df = pd.read_csv("train.csv").fillna("")

tokenizer = Tokenizer(lower=False)
tokenizer.fit_on_texts(qqp_df['question1'])
tokenizer.fit_on_texts(qqp_df['question2'])
is_duplicate = np.asarray(qqp_df['is_duplicate'])

print('Duplicated Rate:', is_duplicate.sum(), '/', len(is_duplicate), '=', is_duplicate.sum() / len(is_duplicate))

print('Vocabulary Sie:', len(tokenizer.word_index))

q1_seq = tokenizer.texts_to_sequences(qqp_df['question1'])
q2_seq = tokenizer.texts_to_sequences(qqp_df['question2'])

seq_len_list = []
for seq in q1_seq:
    seq_len_list.append(len(seq))

for seq in q2_seq:
    seq_len_list.append(len(seq))

print('Min Sequence Length:', min(seq_len_list))
print('Max Sequence Length:', max(seq_len_list))

plt.hist(seq_len_list, density=True, bins=30)
plt.show()

seq_len_array = np.asarray(seq_len_list)

print('Sequence Lengths:')
print('[ 0, 10] = ', (seq_len_array <= 10).sum())
print('(10, 20] = ', ((seq_len_array > 10) & (seq_len_array <= 20)).sum())
print('(20, 30] = ', ((seq_len_array > 20) & (seq_len_array <= 30)).sum())
print('(30, 40] = ', ((seq_len_array > 30) & (seq_len_array <= 40)).sum())
print('(40, 50] = ', ((seq_len_array > 40) & (seq_len_array <= 50)).sum())
print('(50,  +] = ', (seq_len_array > 50).sum())
