import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
import tensorflow.compat.v2 as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import tensorflow.keras as keras
from sklearn.model_selection import StratifiedKFold
import nltk
from nltk import bigrams as nltk_bigrams
from nltk import trigrams as nltk_trigrams


def remove_stop_words(text):
    text = [word for word in text.split() if not word in stop_words]
    text = " ".join(text)
    return text


def remove_punctuation(text):
    text = re.sub(r'[^\w\s]', '', text)
    return text


def vectorize_text(_data, _dict_size: int, _max_length: int):
    max_dictionary_size = _dict_size
    tokenizer = Tokenizer(num_words=max_dictionary_size)
    tokenizer.fit_on_texts(_data['SentimentText'])
    list_tokenized_train = tokenizer.texts_to_sequences(_data['SentimentText'])
    print(f"Max length = {_max_length}")
    X_t = pad_sequences(list_tokenized_train, maxlen=_max_length, padding='post')
    print(len(tokenizer.index_word))
    return X_t


def vectorize_ngrams(_data, _dict_size: int, _max_length: int, ngram_size: int):
    sentences = list()
    for sentence in _data['SentimentText']:
        if ngram_size == 2:
            string_bigrams = nltk_bigrams(sentence.split())
        elif ngram_size == 3:
            string_bigrams = nltk_trigrams(sentence.split())
        else:
            raise ValueError("Not implemented for this ngram size!")
        sentences.append(list(["_".join(bigram) for bigram in string_bigrams]))
    corpus = list()
    for sentence in sentences:
        corpus.append(" ".join(sentence))

    tokenizer = Tokenizer(num_words=_dict_size, filters="")
    tokenizer.fit_on_texts(corpus)
    list_tokenized_train = tokenizer.texts_to_sequences(corpus)
    X_t = pad_sequences(list_tokenized_train, maxlen=_max_length, padding='post')
    len(tokenizer.index_word)
    return X_t


def preprocess_train(data, functions, classifier, _dict_size, word_ngrams=1, words_per_review=None):
    _data = pd.DataFrame(data['SentimentText'])  # Reviews
    y = data['Sentiment']  # Sentiment
    for function in functions:
        _data['SentimentText'] = _data['SentimentText'].apply(lambda x: function(x))
    _row_sizes = _data['SentimentText'].str.split().str.len()
    print(f"Words count: {pd.Series.sum(_row_sizes)}")
    print(_data)
    print(f"Words ngrams: {word_ngrams}")
    # Get longest review (words)
    _data['review_lenght'] = np.array(list(map(lambda x: len(x.split()), _data['SentimentText'])))
    if words_per_review is None:
        max_length = _data['review_lenght'].max()
    else:
        max_length = words_per_review
    # Vectorize reviews
    if word_ngrams == 1:
        X_data = vectorize_text(_data, _dict_size, max_length)
    else:
        X_data = vectorize_ngrams(_data, _dict_size, max_length, word_ngrams)
    result = classifier(X_data, y, max_length)
    return result


def lstm_imdb(_data, _targets, max_length):
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    fold = 0
    results = list()

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                   min_delta=0,
                                                   patience=3,
                                                   verbose=1,
                                                   mode='auto',
                                                   restore_best_weights=True)

    for train, test in kfold.split(_data, _targets):
        print(f"******* Fold {fold + 1} ***********")
        model = keras.models.Sequential([
            keras.layers.Embedding(max_dictionary_size, 16, input_length=max_length),
            keras.layers.Bidirectional(keras.layers.LSTM(16, return_sequences=True)),
            keras.layers.GlobalMaxPooling1D(),
            keras.layers.Dense(16),
            keras.layers.Dense(1, activation="sigmoid")
        ])

        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
        model.fit(_data[train], _targets[train], batch_size=64, epochs=10, verbose=0,
                  validation_data=(_data[test], _targets[test]), callbacks=[early_stopping])
        scores = model.evaluate(_data[test], _targets[test])
        results.append(scores[1])
        fold += 1
    avg = sum(results) / fold * 100
    print(f"Average accuracy = {avg:0.2f} %")
    return avg


def cnn_imdb(_data, _targets, max_length):
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    fold = 0
    results = list()
    filters = 64
    kernel_size = 3

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                   min_delta=0,
                                                   patience=3,
                                                   verbose=1,
                                                   mode='auto',
                                                   restore_best_weights=True)

    for train, test in kfold.split(_data, _targets):
        print(f"******* Fold {fold + 1} ***********")
        model = keras.models.Sequential([
            keras.layers.Embedding(max_dictionary_size, 16, input_length=max_length),
            keras.layers.Conv1D(filters, kernel_size, activation="relu"),
            keras.layers.GlobalMaxPooling1D(),
            keras.layers.Dense(64),
            keras.layers.Activation("relu"),
            keras.layers.Dense(1, activation="sigmoid")
        ])

        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
        model.fit(_data[train], _targets[train], batch_size=32, epochs=10, verbose=0,
                  validation_data=(_data[test], _targets[test]), callbacks=[early_stopping])
        scores = model.evaluate(_data[test], _targets[test])
        results.append(scores[1])
        fold += 1
    avg = sum(results) / fold * 100
    print(f"Average accuracy = {avg:0.2f} %")
    return avg


def save_results(results):
    _path = "yelp_imdb_ex2_results_v2.txt"
    with open(_path, "a") as f:
        for line in results:
            f.write(line)
            f.write("\n")


if __name__ == '__main__':
    print(tf.version.VERSION)
    nltk.download('stopwords')
    stop_words = set(stopwords.words("english"))

    # -----IMDB-----
    # Load data
    path = "imdb_50k.tsv"
    imdbData = pd.read_csv(path, sep='\t', header=0, encoding="utf-8", doublequote=False, escapechar="\\")
    imdbData = imdbData.drop(['id'], axis=1)
    row_sizes = imdbData['SentimentText'].str.split().str.len()
    imdbData['SentimentText'] = imdbData['SentimentText'].str.lower()
    print(f"Words count: {pd.Series.sum(row_sizes)}")
    max_dictionary_size = 10000
    max_review_words = 400
    # Load lemmatized data
    print("Loading lemmatized data")
    path = "Imdb50KLemmatized.tsv"
    imdbDataLem = pd.read_csv(path, sep='\t', header=0, encoding="utf-8", doublequote=False, escapechar="\\")
    imdbDataLem = imdbDataLem.drop(['id'], axis=1)
    row_sizes = imdbDataLem['SentimentText'].str.split().str.len()
    imdbDataLem['SentimentText'] = imdbDataLem['SentimentText'].str.lower()
    print(f"Words count: {pd.Series.sum(row_sizes)}")
    # Remove punctuation
    lstm_result = preprocess_train(imdbData, [remove_punctuation], lstm_imdb, max_dictionary_size, words_per_review=max_review_words)
    cnn_result = preprocess_train(imdbData, [remove_punctuation], cnn_imdb, max_dictionary_size, words_per_review=max_review_words)
    save_results([f"Remove punctuation LSTM: {str(lstm_result)}",
                  f"Remove punctuation CNN: {str(cnn_result)}"])
    # Remove stopwords
    lstm_result = preprocess_train(imdbData, [remove_stop_words], lstm_imdb, max_dictionary_size, words_per_review=max_review_words)
    cnn_result = preprocess_train(imdbData, [remove_stop_words], cnn_imdb, max_dictionary_size, words_per_review=max_review_words)
    save_results([f"Remove stopwords LSTM: {str(lstm_result)}",
                  f"Remove stopwords CNN: {str(cnn_result)}"])
    # Lemmatization
    lstm_result = preprocess_train(imdbDataLem, [], lstm_imdb, max_dictionary_size, words_per_review=max_review_words)
    cnn_result = preprocess_train(imdbDataLem, [], cnn_imdb, max_dictionary_size, words_per_review=max_review_words)
    save_results([f"Lemmatization LSTM: {str(lstm_result)}",
                  f"Lemmatization CNN: {str(cnn_result)}"])
    # Remove stopwords AND remove punctuation
    lstm_result = preprocess_train(imdbData, [remove_stop_words, remove_punctuation], lstm_imdb, max_dictionary_size,
                                   words_per_review=max_review_words)
    cnn_result = preprocess_train(imdbData, [remove_stop_words, remove_punctuation], cnn_imdb, max_dictionary_size,
                                  words_per_review=max_review_words)
    save_results([f"Remove stopwords & punctuation LSTM: {str(lstm_result)}",
                  f"Remove stopwords & punctuation CNN: {str(cnn_result)}"])
    # Remove stopwords AND Lemmatization
    lstm_result = preprocess_train(imdbDataLem, [remove_stop_words], lstm_imdb, max_dictionary_size, words_per_review=max_review_words)
    cnn_result = preprocess_train(imdbDataLem, [remove_stop_words], cnn_imdb, max_dictionary_size, words_per_review=max_review_words)
    save_results([f"Remove stopwords & Lemmatization LSTM: {str(lstm_result)}",
                  f"Remove stopwords & Lemmatization CNN: {str(cnn_result)}"])
