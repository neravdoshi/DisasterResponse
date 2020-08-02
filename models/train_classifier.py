import sys
import pandas as pd
import pandas as pd
import numpy as np
import os
import pickle
import nltk
import re
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, accuracy_score, f1_score, fbeta_score, classification_report
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier


model_pickle_filename = 'disaster_ada_model.pkl'
database_filename = '..DisasterResponse.db'
table_name = 'disaster_message'


def get_df_from_database(database_filename):
    '''
    Return dataframe from the database
    Args:
        database_filename (str): database filename. Default value DATABASE_FILENAME
    Returns:
        df (pandas.DataFrame): dataframe containing the data 
    '''
    engine = create_engine('sqlite:///' + database_filename)
    return pd.read_sql_table(table_name, engine)


def load_data(database_filename):
    """
    Load Data Function
    Arguments:
        database_filename (str):database filename. -> Dafault value of Database_filename
    Output:
        X -> feature DataFrame
        Y -> label DataFrame
        category_names -> used for data visualization (app)
    """
    df = get_df_from_database(database_filename)
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = Y.columns
    return X, Y, category_names

def tokenize(text):
    """
    Tokenization function. 
    Receives as input raw text which afterwards normalized, stop words removed, stemmed and lemmatized.
    Returns tokenized text
    """
    
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    stop_words = stopwords.words("english")
    
    #tokenize
    words = word_tokenize (text)
    
    #stemming
    stemmed = [PorterStemmer().stem(w) for w in words]
    
    #lemmatizing
    words_lemmed = [WordNetLemmatizer().lemmatize(w) for w in stemmed if w not in stop_words]
   
    return words_lemmed


def build_model(grid_search_cv = False):
    '''
    Build the model
    Args:
        grid_search_cv (bool): if True after building the pipeline it will be performed an exhaustive search 
        over specified parameter values ti find the best ones
    Returns:
        pipeline (pipeline.Pipeline): model
    '''
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer = tokenize)), 
                     ('tfidf', TfidfTransformer()), 
                     ('clf', MultiOutputClassifier(RandomForestClassifier()))
                    ])

    #pipeline.get_params()

    if grid_search_cv == True:
        print('Searching for best parameters...')
        parameters = {#'vect__ngram_range': ((1, 1), (1, 2)),
                      #'vect__max_df': (0.5, 0.75, 1.0),
                      #'tfidf__use_idf': (True, False),
                      'clf__estimator__n_estimators': [50, 100, 200],
                      #'clf__estimator__min_samples_split': [2, 3, 4]
        }

        pipeline = GridSearchCV(pipeline, param_grid = parameters)

    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate the model performances and print the results
    Args:
        model (pipeline.Pipeline): model to evaluate
        X_test (pandas.Series): dataset
        Y_test (pandas.DataFrame): dataframe containing the categories
        category_names (str): categories name
    '''
    Y_pred = model.predict(X_test)
    # Calculate the accuracy for each of them.
    for i in range(len(category_names)):
       print('Category: {} '.format(category_names[i]))
       print(classification_report(Y_test.iloc[:, i].values, Y_pred[:, i]))
       print('Accuracy {}\n\n'.format(accuracy_score(Y_test.iloc[:, i].values, Y_pred[:, i])))

def save_model(model, model_pickle_filename):
    '''
    Save in a pickle file the model
    Args:
        model (pipeline.Pipeline): model to be saved
        model_pickle_filename (str): destination pickle filename
    '''
    pickle.dump(model, open(model_pickle_filename, 'wb'))
    

    
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

        print('Saving model...\n    MODEL: {}'.format(model_pickle_filename))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
