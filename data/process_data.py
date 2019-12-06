
"""
Routine Preprocessing Data

Routine to load csv data and save this in database

Script Execution i.e.:
> python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db

Input parameters:
    1) CSV file containing messages (disaster_messages.csv)
    2) CSV file containing categories (disaster_categories.csv)
Output 
    3) SQLite destination database (DisasterResponse.db)
"""

import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Load dataframe from filepaths - messages and categories

    INPUT
    messages_filepath: str, file path of messages data
    categories_filepath: str, file path of categories data

    OUTPUT
    df - pandas DataFrame, merged dataset from messages and categories
    """
    
    # Read message data
    messages = pd.read_csv(messages_filepath)
    # Read categories data
    categories = pd.read_csv(categories_filepath)
    # Merge messages and categories
    df = pd.merge(messages, categories, how='left', on='id')
    
    return df

def clean_data(df):
    """Clean data included in the DataFrame and transform categories part

    INPUT
    df: type pandas DataFrame, Merged dataset from messages and categories

    OUTPUT
    df: type pandas DataFrame, Cleaned datase
    
    """
    # Create a dataframe of the individual category columns
    categories = df['categories'].str.split(pat=';', expand=True)
    # Select the first row of the categories dataframe
    row = categories.loc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    print('Column names:', category_colnames)
    # rename the columns of categories
    categories.columns = category_colnames
    # Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    # Drop the original categories column
    df.drop('categories', axis=1, inplace=True)
    # Concatenate the original dataframe with the new categories dataframe
    df = pd.concat([df, categories], axis=1)
    # drop duplicates
    df.drop_duplicates(inplace=True)
    # Removing entry that is non-binary
    df = df[df['related'] != 2]
    print('Duplicates remaining:', df.duplicated().sum())
    
    return df


def save_data(df, database_filename):
    """Saves DataFrame (df) into sqlite db
    
    INPUT
    df -- type pandas DataFrame
    database_filename -- database filename 

    OUTPUT
    A SQLite database

    """
    name = 'sqlite:///' + database_filename
    engine = create_engine(name)
    conn = engine.connect()
    df.to_sql('Disasters', engine, index=False, if_exists='replace')
    # close the connection
    conn.close()
    engine.dispose()


def main():
    """ Main functions in 3 steps: 
        1. Loads the data
        2. cleans it and 
        3.
        saves it in a database
    """

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