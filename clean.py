import numpy as np
import pandas as pd


def load_df():
    """
        Load our dataframe
    """
    return pd.read_csv("datasets/Seasons_Stats.csv")


def get_year_from_df(df: pd.DataFrame, from_year: int, to_year: int) -> pd.DataFrame:
    """
        Get data for specific years

        Args:
            df -> NBA dataframe with player stats
            from_year -> the year we want data from
            to_year -> the year we want up to

        Returns:
            A dataframe containing the data for the years needed
    """
    return df[(df["Year"] >= from_year) & (df["Year"] <= to_year)]


def get_uniques_only(df: pd.DataFrame) -> pd.DataFrame:
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

    # iterate through all the years and grab only the unique player totals
    for current_year in range(start_year, stop_year + 1):
        current_df = df[df.Year == current_year]
        unique_years.append(current_df.drop_duplicates(subset="Player", keep="first"))

    return pd.concat(unique_years, ignore_index=True)


def get_nba_df(
    unique: bool = True, from_year: int = 2010, to_year: int = 2018
) -> pd.DataFrame:
    """
        Get a copy of the NBA dataframe

        Args:
            unique (default: True) -> Indicator for if we want unique players (most likely always)
            from_year (default: 2010) -> The year we want to start from
            to_year (default: 2017) -> the year we want up till

        Returns:
            A modified version of the nba dataframe
    """
    df = load_df()
    sliced_df = get_year_from_df(df, from_year, to_year)

    if unique:
        return get_uniques_only(sliced_df)

    return sliced_df


if __name__ == "__main__":
    """
        Tester functions for now
    """
    df = load_df()
    sliced_df = get_year_from_df(df, 2010, 2018)
    unique_players = get_uniques_only(sliced_df)
    print(unique_players)
