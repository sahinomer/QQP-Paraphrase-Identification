import datetime

from paraphrase_identificators.siamese_attention_paraphrase_identificator import SiameseAttentionParaphraseIdentificator
from paraphrase_identificators.siamese_paraphrase_identificator import SiameseParaphraseIdentificator

if __name__ == "__main__":
    train = True

    if train:
        identificator = SiameseParaphraseIdentificator()
        identificator.initialize_dataset_frame(path='train.csv', test_rate=0.1)
        identificator.initialize_word_embedding(path='../PassageQueryProject/glove.840B.300d.txt')
        identificator.initialize_model()
        evaluate_score = identificator.train_and_test(path='models/', epochs=4, batch_size=64)
        print(evaluate_score)

    else:
        model_path = 'models/model_' + str(datetime.datetime.now().date())
        identificator = SiameseParaphraseIdentificator()
        identificator.load(path=model_path)

        while True:

            questions = input('Q1+Q2:')
            questions = questions.strip('\"')

            q1 = questions.split('","')[0]
            q2 = questions.split('","')[1]
            print('Q1:', q1)
            print('Q2:', q2)

            is_duplicate = identificator.predict(question1=q1, question2=q2)
            print('is_duplicate:', is_duplicate)
            print('-'*20)
