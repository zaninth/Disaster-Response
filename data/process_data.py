import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Function to load to csv-files and to merge them afterwards

    Args:
        messages_filepath (str): first csv file (messages)
        categories_filepath (str): second csv file (categories)

    Return:
        df (pandas dataframe): Merged messages and categories df, merged on ID
    """

    messages = pd.read_csv(messages_filepath)   # load messages dataset
    categories = pd.read_csv(categories_filepath)   # load categories dataset
    df = messages.merge(categories, how='inner', on='id')  # merge on ID
    return df


def clean_data(df):
    """
    Cleans the data:
        - splits categories into separate columns
        - converts categories values to binary values
        - drops duplicates

    Args:
        df (pandas dataframe): combined categories and messages df
    Returns:
        df (pandas dataframe): Cleaned dataframe with split categories
    """
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories.loc[0]

    # use this row to extract a list of new column names for categories.
    category_colnames = row.apply(lambda x: x[:-2]).tolist()

    # rename the columns of `categories`
    categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1.
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1]
        categories[column] = categories[column].astype('int')

    # drop the original categories column from `df`
    df = df.drop(columns=['categories'], axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1, join='inner')

    # drop duplicates
    df = df.drop_duplicates()

    return df


def save_data(df, database_filename):
    """
    Function:
       Save the Dataframe df in a database
    Args:
       df (DataFrame): A dataframe of messages and categories
       database_filename (str): The file name of the database
       """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')


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
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
