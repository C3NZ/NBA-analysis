import numpy as np
import pandas as pd


def load_df():
    """
        Load our dataframe
    """
    return pd.read_csv("datasets/Season_stats.csv")


def get_year_from_df(df: pd.DataFrame, from_year: int, to_year: int = None):
    """
        Get data for specific years 

        Args:
            df -> NBA dataframe with player stats
            from_year -> the year we want data from
            to_year (default: None) - > the year we want up to

        Returns:
            A dataframe containing the data for the years needed
    """
    if to_year:
        return df[(df["Year"] >= from_year) & (df["Year"] <= to_year)]

    return df[df["Year"] >= from_year]


def get_uniques_only(df: pd.DataFrame):
    '''
        Filter our dataframes for unique players only
    '''
    start_year stop_year = df['Year'].min(), df['Year'].max()
    for j in range(start_year, stop_year):
        current_year = years_to_get[i][1] + j
        current_df = current_decade[current_decade.Year == current_year]
        years.append(current_df.drop_duplicates(subset='Player', keep='first'))
     
    output_df.append(years)
