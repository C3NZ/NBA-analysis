from collections import namedtuple

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from data import get_nba_df, get_train_test


def filter_cols(df: pd.DataFrame) -> tuple:
    """
        Filter unwanted columns from our nba dataframe for our linear regression model

        Args:
            df -> The nba dataframe

        Returns:
            tuple containing (nba player stats, nba player win shares)
    """
    # Columns we want to remove
    unwanted_cols = [
        "Year",
        "Player",
        "Pos",
        "Age",
        "Tm",
        "blanl",
        "blank2",
        "OWS",
        "DWS",
        "WS",
    ]

    # Target column we'd like
    target_col = ["WS"]

    # Grab the nba stats
    nba_stats = df.drop(columns=unwanted_cols)
    nba_ws = df[target_col]

    return nba_stats, nba_ws


# Apply PCA to a dataframe
def apply_pca(df: pd.DataFrame, dimensions: int = 2) -> pd.DataFrame:
    """
        Apply pca to our nba dataframe given the dimensionality 
        we tend to reduce to

        Args:
            df -> The nba dataframe
            dimensions -> the dimensions we tend to reduce

        Returns:
            PCA scaled dataframe

    """
    pca = PCA(n_components=dimensions)
    components = pca.fit_transform(df)

    # Construct our new pca dataframe
    pca_df = pd.DataFrame(
        data=components, columns=["pca-" + str(x + 1) for x in range(dimensions)]
    )
    return pca_df


def create_linear_regression(features: namedtuple, target: namedtuple):
    """
        Create the linear regression model

        Args:
            training_data -> tuple of both training X and Y data
            testing_data -> tuple of both testing X and Y data

        Returns:
            Linear regression model
    """
    pass


def main() -> None:
    """
        Main functionality of our linear regression
    """
    nba_stats, nba_ws = filter_cols(get_nba_df())
    nba_pca = apply_pca(nba_stats.fillna(0), dimensions=3)

    features, target = get_train_test(nba_stats, nba_ws)

    print(features.training)
    print(target.training)


if __name__ == "__main__":
    main()
