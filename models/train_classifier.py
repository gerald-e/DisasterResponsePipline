"""
Routine Train Classifier

Routine load database, train the data and save ML model

Script Execution i.e.:
> python train_classifier.py ../data/DisasterResponse.db classifier.pkl

Input parameters:
    1) SQLite db path (containing pre-processed data)
Output 
    2) Pickle file name to save ML model
"""


import sys
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine
import re
import nltk
import time
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV

nltk.download(['punkt', 'wordnet'])

def load_data(database_filepath):
    '''
    Load data from database as dataframe
    Input:
        database_filepath: File path of sql database
    Output:
        X: Message data (features)
        Y: Categories (target)
        category_names: Labels for all categories used for data visualization app
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Disasters', engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = list(df.columns[4:])

    return X, Y, category_names


def tokenize(text):
    '''
    Clean and tokenize text for modeling. It will replace all non-
    numbers and non-alphabets with a blank space. Next, it will
    split the sentence into word tokens and remove all stopwords.
    The word tokens will then be lemmatized with Nltk's 
    WordNetLemmatizer(), first using noun as part of speech, then verb.
 
    Input:
        text: original message text
    Output:
        clean_tokens: a list containing the cleaned word tokens of the message

    '''
    # replace all non-alphabets and non-numbers with blank space
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize words
    tokens = word_tokenize(text)
    
    # instantiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # instantiate stemmer
    stemmer = PorterStemmer()
    
    clean_tokens = []
    for tok in tokens:
        # lemmtize token using noun as part of speech
        clean_tok = lemmatizer.lemmatize(tok)
        # lemmtize token using verb as part of speech
        clean_tok = lemmatizer.lemmatize(clean_tok, pos='v')
        # stem token
        clean_tok = stemmer.stem(clean_tok)
        # strip whitespace and append clean token to array
        clean_tokens.append(clean_tok.strip())
        
    return clean_tokens


def build_model():
    '''
    Build a ML pipeline using ifidf, random forest, and gridsearch
    Input: None
    Output:
        Results of GridSearchCV
    '''
    # Create pipeline with Classifier
    pipeline = Pipeline([
                        ('vect', CountVectorizer(tokenizer=tokenize)),
                        ('tfidf', TfidfTransformer()),
                        ('clf', MultiOutputClassifier(RandomForestClassifier()))
                        ])

    # params dict to tune a model
    # parameters = {
    #     'clf__estimator__min_samples_split': [2, 4],
    #     'clf__estimator__max_features': [None, 'log2', 'sqrt'],
    #     'clf__estimator__criterion': ['gini', 'entropy'],
    #     'clf__estimator__max_depth': [25, 100, 200],
    # }
    parameters = {
        'clf__estimator__min_samples_split': [2, 4],
        'clf__estimator__max_features': [None, 'log2', 'sqrt'],
        'clf__estimator__criterion': ['gini', 'entropy'],
        'clf__estimator__max_depth': [10, 50, None],
    }
    # instantiate a gridsearchcv object with the params defined
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=4, n_jobs=8)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate model performance using test data
    Input: 
        model: Model to be evaluated
        X_test: Test data (features)
        Y_test: True lables for Test data
        category_names: Labels for 36 categories
    Output:
        Print accuracy and classfication report for each category
    '''
    # Get results and add them to a dataframe.
    Y_pred = model.predict(X_test)
    
    # Calculate the accuracy for each of them.
    for i in range(len(category_names)):
        print("Category:", 
              category_names[i],"\n", 
              classification_report(Y_test.iloc[:, i].values, Y_pred[:, i]))
        print('Accuracy of %25s: %.2f' %(category_names[i], 
                                         accuracy_score(Y_test.iloc[:, i].values, 
                                                        Y_pred[:,i])))


def save_model(model, model_filepath):
    '''
    Save model as a pickle file 
    Input: 
        model: Model to be saved
        model_filepath: path of the output pick file
    Output:
        A pickle file of saved model
    '''
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        starttime = time.time()
        print('Training model...')
        model.fit(X_train.as_matrix(), Y_train.as_matrix())
        runtime = time.time() - starttime
        print('Function completed in {:.0f}m {:.0f}s'.format(runtime // 60, runtime % 60))
        
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