
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


class QQPDataFrame:

    def __init__(self, path=None):
        self.seq_len = 50
        self.qqp_df_train = None
        self.qqp_df_test = None

        if path is not None:
            self.qqp_df = pd.read_csv(path).fillna("")

        self.tokenizer = Tokenizer(lower=False)

    def split_train_test(self, test_rate):
        total_sample = len(self.qqp_df)
        train_sample = round(total_sample * (1-test_rate))
        self.qqp_df_train = self.qqp_df[:train_sample]
        self.qqp_df_test = self.qqp_df[train_sample:]

        print('Number of total instances : %6d' % total_sample)
        print('Number of train instances : %6d' % train_sample)
        print('Number of test instances  : %6d' % (total_sample - train_sample))

    def fit_tokenizer(self):
        print('Fit tokenizer on train instances...', end='', flush=True)
        self.tokenizer.fit_on_texts(self.qqp_df_train['question1'])
        self.tokenizer.fit_on_texts(self.qqp_df_train['question2'])
        print('done')

    def get_train_data(self):
        question1, question2 = self.get_as_sequence(data_frame=self.qqp_df_train)
        return question1, question2, self.qqp_df_train['is_duplicate']

    def get_test_data(self):
        question1, question2 = self.get_as_sequence(data_frame=self.qqp_df_test)
        return question1, question2, self.qqp_df_test['is_duplicate']

    def get_prediction_data(self, question1, question2):
        data_frame = pd.DataFrame({'question1': [question1], 'question2': [question2]})
        return self.get_as_sequence(data_frame=data_frame)

    def get_as_sequence(self, data_frame):
        question1 = self.tokenizer.texts_to_sequences(data_frame['question1'])
        question1 = pad_sequences(question1, maxlen=self.seq_len)

        question2 = self.tokenizer.texts_to_sequences(data_frame['question2'])
        question2 = pad_sequences(question2, maxlen=self.seq_len)

        return question1, question2
