import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load data from filepath
    ----------
    Parameters:
    - messages_filepath: file path where messages data is stored
    - categories_filepath: file path where category data is stored
    ----------
    Return: merged dataframe
    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')

    return df


def clean_data(df):
    """
    Clean data by creating columns for all labels
    ----------
    Parameters:
    - df: dataframe
    ----------
    Return: dataframe
    """

    # create a dataframe of the 36 individual category columns and rename columns 
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames

    # convert category values to just numbers 0 or 1
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x[-1])
        categories[column] = categories[column].astype(int)

    # replace categories column in df with new category columns
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)

    # remove duplicates and convert multiclass to binary
    df.drop_duplicates(inplace=True)
    df = df[df['related'] != 2] # 4 instances have the value 2. Model doesn't work with these, so I remove them

    return df


def save_data(df, database_filepath):
    """
    Save data to sqlite database
    ----------
    Parameters:
    df: dataframe
    database_filename: file path where data should be saved
    ----------
    Return: /
    """

    engine = create_engine(f'sqlite:///{database_filepath}')
    df.to_sql('categorized_messages', engine, index=False, if_exists='replace')  


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