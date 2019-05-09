import numpy as np
import pandas as pd


def load_df():
    """
        Load our dataframe
    """
    return pd.read_csv("datasets/Seasons_Stats.csv")


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
    """
        Filter our dataframes for unique players only

        Args:
            df -> The NBA dataframe with all the players 

        Returns:
            A dataframe containing only unique players
    """
    unique_df = pd.DataFrame()
    unique_years = []
    start_year, stop_year = int(df["Year"].min()), int(df["Year"].max())
    for current_year in range(start_year, stop_year):
        current_df = df[df.Year == current_year]
        unique_years.append(current_df.drop_duplicates(subset="Player", keep="first"))

    return pd.concat(unique_years, ignore_index=True)


if __name__ == "__main__":
    df = load_df()
    sliced_df = get_year_from_df(df, 2010, 2017)
    unique_players = get_uniques_only(sliced_df)
