import sys
from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

import pandas as pd
import re
import pickle

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report


def load_data(database_filepath):
    """ Load messages and categories from a SQLite database
        and returns feature and target variables X and Y
    
    Args:
        database_filepath: String. Filepath of database
    
    Returns:
        X: Dataframe. Feature-Dataframe with messages
        Y: Dataframe. Target-Dataframe with categories
        Y.keys(): List. List of column names for Y-Dataframe
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('messages_and_categories', engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    return X, Y, Y.keys()


def tokenize(text):
    """ Tokenization function for processing the text data
        with the following steps
        # Normalize text
        # Tokenize text
        # Remove stop words
        # Reduce words to their root form
        # Lemmatize verbs by specifying pos
        
    Args:
        text: String. One message
    
    Returns:
        lemmed: List. Tokenized and lemmed list of words
    """
    
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    words = text.split() 
    words = [w for w in words if w not in stopwords.words("english")]
    
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    lemmed = [WordNetLemmatizer().lemmatize(w, pos='v') for w in lemmed]
    
    return lemmed


def build_model():
    """ Build a machine learning pipeline in Pipeline Format 
        with CountVectorizer as input and a MultiOutputClassifier 
        with a RandomForestClassifier as output
        
    Args:
    
    Returns:
        pipeline: Pipeline. Machine learning pipeline
    """
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf',  MultiOutputClassifier(RandomForestClassifier(n_estimators=100)))
    ])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """ Evaluates the model with a testing Dataset. Report the f1 score, 
    precision and recall for each output category of the dataset on the console
        
    Args:
        model: Model. Trained model
        X_test: Array. Testing set of messages
        Y_test: Array. Testing set of correct categories
        category_names: List. Names of the columns for output
    Returns:
   
    """
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names))
    pass


def save_model(model, model_filepath):
    """ Save the model to a pickle file
    
    Args:
        model: Model. Model to save
        model_filepath: String. Filepath of pickle file
    
    Returns:

    """
    temp = open(model_filepath,'wb')
    pickle.dump(model, temp)
    temp.close()
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

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()