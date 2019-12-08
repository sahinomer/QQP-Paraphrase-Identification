
from embedding import WordEmbedding
from qqp_dataframe import QQPDataFrame


class ParaphraseIdentificator:

    def __init__(self):
        self.model_name = None
        self.qqp_df = None
        self.word_embedding = None
        self.model = None

    def initialize_dataset_frame(self, path, test_rate=0.1):
        self.qqp_df = QQPDataFrame(path=path)
        self.qqp_df.split_train_test(test_rate=test_rate)
        self.qqp_df.fit_tokenizer()

    def initialize_word_embedding(self, path):
        self.word_embedding = WordEmbedding(embfile=path)
        self.word_embedding.create_embedding_matrix(self.qqp_df.tokenizer)

    def train_and_test(self,  path, epochs=10, batch_size=64):
        self.train(epochs=epochs, batch_size=batch_size)
        self.save(path)
        del self.model
        self.load(path)
        return self.test(batch_size=batch_size)

    def initialize_model(self):
        pass

    def train(self, epochs=10, batch_size=64):
        pass

    def test(self, batch_size=64):
        pass

    def predict(self, question1, question2):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass
