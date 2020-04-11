import sys
import nltk
nltk.download(['punkt', 'wordnet'])
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
import os
import pickle

def load_data(database_filepath):
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('assignment',engine)
    X = df.message.values
    Y = df.drop(['id','message','original','genre'],axis=1).values
    category_names = df.drop(['id','message','original','genre'],axis=1).columns
    return X,Y,category_names


def tokenize(text):
    r_text = text.replace('.', "").replace(',',"").replace(':',"")
    tokens = word_tokenize(r_text)
    lemmatizer = WordNetLemmatizer()
    clean_data = []
    for t in tokens:
        clean_t = lemmatizer.lemmatize(t).lower().strip()
        clean_data.append(clean_t)
    return clean_data


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(KNeighborsClassifier()))
    ])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    for i,colName in enumerate(category_names):
        print(colName, classification_report(Y_test[i], y_pred[i], target_names=['0','1']))
    pass


def save_model(model, model_filepath):
    word_dict_path = os.path.join(model_filepath)   
    with open(word_dict_path, "wb") as f:
      pickle.dump(model, f)
    pass


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

#         print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()