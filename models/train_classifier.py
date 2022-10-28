from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer, classification_report
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import pickle
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
import re
import warnings
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine
import sys
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
warnings.simplefilter('ignore')
nltk.download('stopwords', 'punkt', 'wordnet', 'averaged_perceptron_tagger')


def load_data(database_filepath):
    """
    Load Data Function

    Arguments:
        database_filepath -> path to SQLite db
    Output:
        X -> feature DataFrame
        Y -> label DataFrame
        category_names -> used for data visualization (app)
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df['message']  # Message column
    Y = df.iloc[:, 4:]  # Classification label
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    """
    Function: split text into words and return the root form of the words, and removing 
    Args:
      text(str): the message
    Return:
      clean_tokens - tokens after cleaning, tokenization and lemmatization of the text
    """
    # Detecting URLs and removing punctuation
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # Normalize, tokenize and lemmatize
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Creates a model using Pipeline and GridSearchCV 
    Args:
        None
    Returns:
        cv (scikit-learn GridSearchCV): Grid search model object
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))

    ])

    parameters = {
        'vect__max_features': (None, 5000),
        'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [50, 100],
        'clf__estimator__min_samples_split': [2, 3],
    }

    model = GridSearchCV(pipeline, param_grid=parameters,
                         cv=2, n_jobs=-1, verbose=3)

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluate model
    Input:
        model: sklearn.model_selection.GridSearchCV.  It contains a sklearn estimator.
        X_test: numpy.ndarray. Disaster messages.
        Y_test: numpy.ndarray. Disaster categories for each messages
        category_names: Disaster category names.
    """
    Y_pred = model.predict(X_test)

    print(classification_report(Y_test.iloc[:, 1:].values, np.array(
        [x[1:] for x in Y_pred]), target_names=category_names))
    accuracy = (Y_pred == Y_test.values).mean()
    print(f'The model accuracy is {accuracy:.3f}')


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
