import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
import tensorflow.compat.v2 as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import tensorflow.keras as keras
from gensim.models.phrases import Phrases, Phraser
import nltk


class Dataset:
    def __init__(self):
        self.df_train = None
        self.df_test = None
        self.df_valid = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.x_valid = None
        self.y_valid = None

    def copy(self):
        _dataset = Dataset()
        _dataset.df_train = pd.DataFrame(self.df_train['SentimentText'])  # Reviews
        _dataset.df_test = pd.DataFrame(self.df_test['SentimentText'])
        _dataset.df_valid = pd.DataFrame(self.df_valid['SentimentText'])
        _dataset.y_train = self.df_train['Sentiment']  # Sentiment
        _dataset.y_test = self.df_test['Sentiment']
        _dataset.y_valid = self.df_valid['Sentiment']
        return _dataset


def remove_stop_words(text):
    text = [word for word in text.split() if not word in stop_words]
    text = " ".join(text)
    return text


def remove_punctuation(text):
    text = re.sub(r'[^\w\s]', '', text)
    return text


# tokenization + padding
def vectorize_text(_data, _dict_size: int, _max_length: int):
    max_dictionary_size = _dict_size
    tokenizer = Tokenizer(num_words=max_dictionary_size)
    tokenizer.fit_on_texts(_data.df_train['SentimentText'])
    list_tokenized_train = tokenizer.texts_to_sequences(_data.df_train['SentimentText'])
    print(f"Max length = {_max_length}")
    # train data
    x_train = pad_sequences(list_tokenized_train, maxlen=_max_length, padding='post')
    print(len(tokenizer.index_word))
    _data.x_train = x_train
    # test data
    list_tokenized_train = tokenizer.texts_to_sequences(_data.df_test['SentimentText'])
    x_test = pad_sequences(list_tokenized_train, maxlen=_max_length, padding='post')
    _data.x_test = x_test
    # validation data
    list_tokenized_train = tokenizer.texts_to_sequences(_data.df_valid['SentimentText'])
    x_valid = pad_sequences(list_tokenized_train, maxlen=_max_length, padding='post')
    _data.x_valid = x_valid
    return _data


def join_phrases(text: str, ngrams):
    return " ".join(ngrams[text.split()])


def vectorize_ngrams(_data, _dict_size: int, _max_length: int, ngram_size: int, threshold: int):
    all_reviews = _data.df_train['SentimentText'].values
    all_reviews = np.array(list(map(lambda x: x.split(), all_reviews)))

    ngrams = Phrases(sentences=all_reviews, threshold=threshold)
    if ngram_size == 3:
        ngrams = Phrases(sentences=ngrams[all_reviews])
    elif ngram_size > 3:
        raise ValueError("Not implemented for this ngram size!")
    phraser = Phraser(ngrams)

    text_ngrams_train = _data.df_train['SentimentText'].apply(lambda x: join_phrases(x, phraser))
    text_ngrams_valid = _data.df_valid['SentimentText'].apply(lambda x: join_phrases(x, phraser))
    text_ngrams_test = _data.df_test['SentimentText'].apply(lambda x: join_phrases(x, phraser))
    tokenizer = Tokenizer(num_words=max_dictionary_size, filters='!"#$%&()*+,-./:;<=>?@[\\]^`{|}~\t\n')
    tokenizer.fit_on_texts(text_ngrams_train)
    list_tokenized_train = tokenizer.texts_to_sequences(text_ngrams_train)
    _data.x_train = pad_sequences(list_tokenized_train, maxlen=_max_length, padding='post')

    list_tokenized_valid = tokenizer.texts_to_sequences(text_ngrams_valid)
    _data.x_valid = pad_sequences(list_tokenized_valid, maxlen=_max_length, padding='post')

    list_tokenized_test = tokenizer.texts_to_sequences(text_ngrams_test)
    _data.x_test = pad_sequences(list_tokenized_test, maxlen=_max_length, padding='post')
    return _data


# text preprocessing + model training + evaluation
def preprocess_train(dataset, functions, classifier, _dict_size, word_ngrams=1, words_per_review=None, threshold=10):
    _dataset = dataset.copy()

    for function in functions:
        _dataset.df_train['SentimentText'] = _dataset.df_train['SentimentText'].apply(lambda x: function(x))
        _dataset.df_test['SentimentText'] = _dataset.df_test['SentimentText'].apply(lambda x: function(x))
        _dataset.df_valid['SentimentText'] = _dataset.df_valid['SentimentText'].apply(lambda x: function(x))
    _row_sizes = _dataset.df_train['SentimentText'].str.split().str.len()
    print(f"Words count: {pd.Series.sum(_row_sizes)}")
    print(_dataset.df_train)
    print(f"Words ngrams: {word_ngrams}")
    # Set max review length (words)
    max_length = words_per_review
    if words_per_review is None:
        raise ValueError("Set parameter words_per_review!")
    # Vectorize reviews
    if word_ngrams == 1:
        _dataset = vectorize_text(_dataset, _dict_size, max_length)
    else:
        _dataset = vectorize_ngrams(_dataset, _dict_size, max_length, word_ngrams, threshold)
    result = classifier(_dataset, max_length, _dict_size)
    return result


def lstm(_data, max_length, dict_size):
    model = keras.models.Sequential([
        keras.layers.Embedding(max_dictionary_size, 16, input_length=max_length),
        keras.layers.Bidirectional(keras.layers.LSTM(16, return_sequences=True)),
        keras.layers.GlobalMaxPooling1D(),
        keras.layers.Dense(16),
        keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   min_delta=0,
                                                   patience=4,
                                                   verbose=1,
                                                   mode='auto',
                                                   restore_best_weights=True)
    model.fit(_data.x_train, _data.y_train, batch_size=128, epochs=12, validation_data=(_data.x_valid, _data.y_valid), callbacks=[early_stopping])
    test_loss, test_acc = model.evaluate(_data.x_test, _data.y_test)
    print('Test accuracy: ', test_acc)
    return test_acc


def cnn(_data, max_length, dict_size):
    filters = 64
    kernel_size = 3

    model = keras.models.Sequential([
        keras.layers.Embedding(max_dictionary_size, 16, input_length=max_length),
        keras.layers.Conv1D(filters, kernel_size, activation="relu"),
        keras.layers.GlobalMaxPooling1D(),
        keras.layers.Dense(64),
        keras.layers.Activation("relu"),
        keras.layers.Dense(1, activation="sigmoid")
    ])

    model.summary()
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   min_delta=0,
                                                   patience=4,
                                                   verbose=1,
                                                   mode='auto',
                                                   restore_best_weights=True)
    model.fit(_data.x_train, _data.y_train, batch_size=128, epochs=12, validation_data=(_data.x_valid, _data.y_valid), callbacks=[early_stopping])
    test_loss, test_acc = model.evaluate(_data.x_test, _data.y_test)
    print('Test accuracy: ', test_acc)
    return test_acc


def save_results(results):
    _path = "amazon_ex2_results.txt"
    with open(_path, "a") as f:
        for line in results:
            f.write(line)
            f.write("\n")


if __name__ == '__main__':
    # basic configuration
    print(tf.version.VERSION)
    nltk.download('stopwords')
    stop_words = set(stopwords.words("english"))
    max_dictionary_size = 10000
    max_review_words = 180

    # -----Amazon-----
    # Load data
    data = Dataset()
    # # train data
    print("Loading data")
    path = "../data/AmazonTrainSet1M.tsv"
    data_train = pd.read_csv(path, sep='\t', header=0, encoding="utf-8")
    data_train['SentimentText'] = data_train['SentimentText'].str.lower()
    data.df_train = data_train
    # # test data
    path = "../data/AmazonTestSet400k2.tsv"
    data_test = pd.read_csv(path, sep='\t', header=0, encoding="utf-8")
    data_test['SentimentText'] = data_test['SentimentText'].str.lower()
    data.df_test = data_test
    # # validation data
    path = "../data/AmazonValidationSet100K.tsv"
    data_valid = pd.read_csv(path, sep='\t', header=0, encoding="utf-8")
    data_valid['SentimentText'] = data_valid['SentimentText'].str.lower()
    data.df_valid = data_valid
    # Load lemmatized data
    data_lem = Dataset()
    # # train data
    print("Loading lemmatized data")
    path = "../data/AmazonTrainLemmatized.tsv"
    lem_train = pd.read_csv(path, sep='\t', header=0, encoding="utf-8")
    lem_train['SentimentText'] = lem_train['SentimentText'].str.lower()
    data_lem.df_train = lem_train
    # # test data
    path = "../data/AmazonTestLemmatized.tsv"
    lem_test = pd.read_csv(path, sep='\t', header=0, encoding="utf-8")
    lem_test['SentimentText'] = lem_test['SentimentText'].str.lower()
    data_lem.df_test = lem_test
    # # validation data
    path = "../data/AmazonValidationLemmatized.tsv"
    lem_valid = pd.read_csv(path, sep='\t', header=0, encoding="utf-8")
    lem_valid['SentimentText'] = lem_valid['SentimentText'].str.lower()
    data_lem.df_valid = data_valid

    # Remove punctuation
    print("Remove punctuation 1/5")
    lstm_result = preprocess_train(data, [remove_punctuation], lstm, max_dictionary_size, words_per_review=max_review_words)
    cnn_result = preprocess_train(data, [remove_punctuation], cnn, max_dictionary_size, words_per_review=max_review_words)
    save_results([f"Remove punctuation LSTM: {str(lstm_result)}",
                  f"Remove punctuation CNN: {str(cnn_result)}"])
    # Remove stopwords
    print("Remove stopwords 2/5")
    lstm_result = preprocess_train(data, [remove_stop_words], lstm, max_dictionary_size, words_per_review=max_review_words)
    cnn_result = preprocess_train(data, [remove_stop_words], cnn, max_dictionary_size, words_per_review=max_review_words)
    save_results([f"Remove stopwords LSTM: {str(lstm_result)}",
                  f"Remove stopwords CNN: {str(cnn_result)}"])
    # Lemmatization
    print("Lemmatization 3/5")
    lstm_result = preprocess_train(data_lem, [], lstm, max_dictionary_size, words_per_review=max_review_words)
    cnn_result = preprocess_train(data_lem, [], cnn, max_dictionary_size, words_per_review=max_review_words)
    save_results([f"Lemmatization LSTM: {str(lstm_result)}",
                  f"Lemmatization CNN: {str(cnn_result)}"])
    # Remove stopwords AND remove punctuation
    print("Remove stopwords and remove punctuation 4/5")
    lstm_result = preprocess_train(data, [remove_stop_words, remove_punctuation], lstm, max_dictionary_size,
                                   words_per_review=max_review_words)
    cnn_result = preprocess_train(data, [remove_stop_words, remove_punctuation], cnn, max_dictionary_size,
                                  words_per_review=max_review_words)
    save_results([f"Remove stopwords & punctuation LSTM: {str(lstm_result)}",
                  f"Remove stopwords & punctuation CNN: {str(cnn_result)}"])
    # Remove stopwords AND Lemmatization
    print("Remove stop and Lemmatization 5/5")
    lstm_result = preprocess_train(data_lem, [remove_stop_words], lstm, max_dictionary_size, words_per_review=max_review_words)
    cnn_result = preprocess_train(data_lem, [remove_stop_words], cnn, max_dictionary_size, words_per_review=max_review_words)
    save_results([f"Remove stopwords & Lemmatization LSTM: {str(lstm_result)}",
                  f"Remove stopwords & Lemmatization CNN: {str(cnn_result)}"])

    # N-ngrams
    results = []
    for i in [2, 3]:
        for j in [10, 20, 40]:
            print(f"LTSM ngram {i}, threshold {j}")
            scores = preprocess_train(data, [], lstm, max_dictionary_size, words_per_review=max_review_words,
                                      word_ngrams=i, threshold=j)
            results.append(f"Lemmatization LSTM with ngram {i} threshold {j}: {str(scores)}")

    for i in [2, 3]:
        for j in [10, 20, 40]:
            print(f"CNN ngram {i}, threshold {j}")
            scores = preprocess_train(data, [], cnn, max_dictionary_size, words_per_review=max_review_words,
                                      word_ngrams=i, threshold=j)
            results.append(f"Lemmatization CNN with ngram {i} threshold {j}: {str(scores)}")

    save_results(results)
