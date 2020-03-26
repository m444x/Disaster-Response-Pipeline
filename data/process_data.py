import sys
import pandas as pd
from sqlalchemy import create_engine
    
def load_data(messages_filepath, categories_filepath):
    """ Load messages and categories from 2 CSV Files
        and merge them in one Dataframe by ID.
    
    Args:
        messages_filepath: CSV File. Filepath of Messages
        categories_filepath: CSV File. Filepath of Categories
    
    Returns:
        df: Dataframe. Merged Dataframe with messages and categories
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')
    return df


def clean_data(df):
    """ Cleans a Dataframe with the following steps:
        - Split categories into separate category columns.
        - Convert category values to just numbers 0 or 1.
        - Replace categories column with new category columns.
        - Remove duplicates
    
    Args:
        df: Dataframe. Merged Dataframe with messages and categories
    
    Returns:
        df: Dataframe. Cleaned Datframe with added columns for each categorie
    """
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = row.str.split('-').str.get(0)
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].str.split('-').str.get(1)
        categories[column] = pd.to_numeric(categories[column])
    df.drop(['categories'], axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    
    df.drop_duplicates(inplace=True)
    
    df.related.replace(2,1,inplace=True)
    
    return df
         


def save_data(df, database_filename):
    """ Saves the dataset into an sqlite database.
    
    Args:
        df: Dataframe. Cleaned Dataframe with messages and categories
        database_filename: String. Filename for database
    
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('messages_and_categories', engine, index=False, if_exists='replace') 


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()