from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from keras.optimizers import SGD, Adam
from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import pandas as pd
import re
import os


def clean_text(text):
    """
clean_text(text: string)

clean the input text by removing all non-english-alphabet characters, and removing stopwords

returns cleaned text
    """
    clean_text = text.lower()
    regex = re.compile('[^a-zA-Z ]')
    clean_text = regex.sub(' ', clean_text)
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(clean_text)
    filtered_text = [w for w in word_tokens if not w in stop_words]
    clean_text = ' '.join(filtered_text)
    return clean_text


def create_model(input_dim, n_classes):
    """
    create_model(input_shape, n_classes)
    model creation resides here, modify this function to modify model structure

    """

    model = Sequential()
    model.add(Dense(200, activation='relu', input_dim=input_dim))
    model.add(Dropout(0.3))
    model.add(Dense(150, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(n_classes, activation='sigmoid'))

    adam = Adam(lr=0.01, decay=1e-6)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['acc'])

    return model


CSV_PATH = os.path.join(os.path.dirname(__file__), "train.csv")

if __name__ == "__main__":

    data = pd.read_csv(CSV_PATH, index_col=0)
    data["clean_comment"] = data.comment_text.apply(clean_text)

    vectorizer = TfidfVectorizer(max_features=50000)
    X = vectorizer.fit_transform(data.clean_comment)

    not_class_cols = ['comment_text', 'clean_comment']
    Y = data.drop(not_class_cols, axis=1)

    prediction_series = Y.columns

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.1)

    model = create_model(x_train.shape[1], y_train.shape[1])

    model.fit(x_train, y_train, epochs=2, batch_size=100, validation_split=.1)

    val_loss, val_acc = model.evaluate(x_test, y_test)

    print("Validation loss:", val_loss, "\t", "Validation Accuracy", val_acc)

    model.save("dense.h5")

    pickle.dump(vectorizer, open("tfidf.pickle", "wb"))

    pickle.dump(prediction_series, open("labels.pickle", "wb"))

    print("Done Saving model and vectorizer and labels")
    # model_json = model.to_json()
    # with open("dense.json", "w") as json_file:
    #     json_file.write(model_json)
    # model.save_weights("dense-weights.h5")
    # print("Saved model to disk")
