import re
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer


class Preprocessor:

    def __init__(self):
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.wordnet_lemmatizer = WordNetLemmatizer()
        self.snowball_stemmer = SnowballStemmer('english')

    def preprocess_text(self, texts):
        """
        Preprocess text:
            Lowercase
            Lemmatize
            Stemming
        """
        preprocessed = list()

        for text in texts:

            # acronym
            text = re.sub(r"can\'t", "can not", text)
            text = re.sub(r"cannot", "can not ", text)
            text = re.sub(r"what\'s", "what is", text)
            text = re.sub(r"What\'s", "what is", text)
            text = re.sub(r"\'ve ", " have ", text)
            text = re.sub(r"n\'t", " not ", text)
            text = re.sub(r"i\'m", "i am ", text)
            text = re.sub(r"I\'m", "i am ", text)
            text = re.sub(r"\'re", " are ", text)
            text = re.sub(r"\'d", " would ", text)
            text = re.sub(r"\'ll", " will ", text)
            text = re.sub(r"\'s", " ", text)

            word_tokens = self.tokenizer.tokenize(text)

            words = list()
            for word in word_tokens:
                word = word.lower()
                word = self.wordnet_lemmatizer.lemmatize(word)
                word = self.snowball_stemmer.stem(word)

                words.append(word)

            preprocessed.append(' '.join(words))

        return preprocessed
