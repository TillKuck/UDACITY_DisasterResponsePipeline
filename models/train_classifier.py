import sys
import numpy as np
import pandas as pd
import re
from sqlalchemy import create_engine
import pickle

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


def load_data(database_filepath):
    """
    Load data from a sqlite
    ----------
    Parameters:
    - database_filepath: file path where data is stored
    ----------
    Return: Fitted model
    """

    engine = create_engine('sqlite:///', database_filepath)
    df = pd.read_sql_table('categorized_messages', engine)
    df = df[df['related'] != 2] # 4 instances have the value 2. Model doesn't work with these, so I remove them
    X = df['message'].head(5000) # For faster processing, I limit the data
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1).head(5000)
    
    return X, Y


def tokenize(text):
    """
    Transform messages into tokens by applying several steps.
    1)remove punctuation 2)tokenize messages and lowercase them
    3)remove stopwords 4) lemmatize tokens
    ----------
    Parameters:
    - text: Text (str) that should be tokenized
    ----------
    Return: a list of fitted tokens
    """

    clean_tokens = []
    for message in text:
        message = re.sub(r'[^a-zA-Z0-9]', ' ', message).strip()
        tokens = word_tokenize(message.lower())
        tokens = [token for token in tokens if token not in stopwords.words("english")]
        tokens = [WordNetLemmatizer().lemmatize(token) for token in tokens]
#         tokens = [PorterStemmer().stem(token) for token in tokens]
        clean_tokens.append(' '.join(tokens))
        
    return clean_tokens


def build_model():
    """
    Build machine learning model by using pipeline to transform data,
    and train random forest model utilizing grid search
    ----------
    Parameters:
    ----------
    Return: Fitted model
    """

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'clf__estimator__min_samples_leaf': [5],
        'clf__estimator__max_depth': [6],
        'clf__estimator__n_estimators': [10, 20]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate machine learning model on test set
    ----------
    Parameters:
    - model: The machine learning model to be saved
    - X_test: The features of the test set (dataframe)
    - Y_test: The labels of the test set (dataframe)
    - category_names: Column names of Y_test
    ----------
    Return: Print classification report with precision, recall and f1-score
    """

    y_pred_cv = model.predict(X_test)
    print(classification_report(Y_test, y_pred_cv, target_names=category_names))


def save_model(model, model_filepath):
    """
    Save a machine learning model as pickle file
    ----------
    Parameters:
    - model: The machine learning model to be saved
    - model_filepath: The file path where the model should be saved
    ----------
    Return: Print where model is saved
    """

    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)
    print(f'Model saved as pickl file at: {model_filepath}')


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
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
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()