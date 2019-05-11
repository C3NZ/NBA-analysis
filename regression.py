import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from clean import get_nba_df


def filter_cols(df: pd.DataFrame) -> tuple:
    """
        Filter unwanted columns from our nba dataframe for our linear regression model

        Args:
            df -> The nba dataframe

        Returns:
            tuple containing (nba player stats, nba player win shares)
    """
    # Columns we want to remove
    unwanted_cols = ["Year", "Pos", "Age", "Tm", "blanl", "blank2", "OWS", "DWS", "WS"]

    # Target column we'd like
    target_col = ["WS"]

    # Grab the nba stats
    nba_stats = df.drop(columns=unwanted_cols)
    nba_ws = df[target_col]

    return nba_stats, nba_ws


# Apply PCA to a dataframe
def apply_pca(df: pd.DataFrame, dimensions: int = 2):
    """
        Apply pca to our nba dataframe

    """
    pca = PCA(n_components=dimensions)
    components = pca.fit_transform(df)

    pca_df  = pd.DataFrame(data=components, columns==['pca-' + str(x + 1) for x in range(dimensions)])
    return pca_df

def main() -> None:
    """
        Main functionality of our linear regression
    """
    nba_stats, nba_ws = filter_cols(get_nba_df())
    print(nba_stats)
    print(nba_ws)


if __name__ == "__main__":
    main()
