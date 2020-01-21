
from paraphrase_identificators.siamese_attention_paraphrase_identificator import SiameseAttentionParaphraseIdentificator
from paraphrase_identificators.siamese_paraphrase_identificator import SiameseParaphraseIdentificator
from paraphrase_identificators.wmd_paraphrase_identificator import WordMoverDistanceParaphraseIdentificator

if __name__ == "__main__":
    train = True
    model = ['wmd', 'siamese', 'siamese+attention'][2]

    if train:
        if model == 'wmd':
            identificator = WordMoverDistanceParaphraseIdentificator()
            identificator.initialize_dataset_frame(path='train.csv', test_rate=0.1)
            identificator.initialize_model()
            evaluate_score = identificator.test()
        else:
            identificator = None
            if model == 'siamese':
                identificator = SiameseParaphraseIdentificator()

            elif model == 'siamese+attention':
                identificator = SiameseAttentionParaphraseIdentificator()

            identificator.initialize_dataset_frame(path='train.csv', test_rate=0.1)
            identificator.initialize_word_embedding(path='../PassageQueryProject/glove.840B.300d.txt')
            identificator.initialize_model()
            identificator.model.summary()
            evaluate_score = identificator.train_and_test(path='models/', epochs=8, batch_size=64)

        print(evaluate_score)

    else:
        model_path = 'models/siamese_attention_dnn+8epoch+%87_2020-01-21'
        identificator = SiameseAttentionParaphraseIdentificator()
        identificator.load(path=model_path)

        while True:

            questions = input('Q1+Q2:')
            questions = questions.strip('\"')

            q1 = questions.split('","')[0]
            q2 = questions.split('","')[1]
            print('Q1:', q1)
            print('Q2:', q2)

            similarity = identificator.predict(question1=q1, question2=q2)[0, 0]

            print('Similarity:', similarity)
            score_bar = '#' * int(round(similarity * 100)) + '-' * (100 - int(round(similarity * 100)))
            score_bar = score_bar[:50] + '|' + score_bar[50:]
            print('[%s]   %0.2f%%' % (score_bar, similarity*100))

            print('\n' + '-'*120 + '\n')
