from collections import namedtuple

import pandas as pd
from data import get_nba_df, get_train_test
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler


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
        "Unnamed: 0",
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


def apply_scaling(features: pd.DataFrame, scale_type: str = "Standard") -> pd.DataFrame:
    """
        apply scaling to our dataframe 
    """
    scaled_features = features
    if scale_type == "Standard":
        std_scaler = StandardScaler()
        std_features = std_scaler.fit_transform(features)
        scaled_features = pd.DataFrame(std_features, columns=scaled_features.columns)

    elif scale_type == "MinMax":
        mm_scaler = MinMaxScaler()
        mm_features = mm_scaler.fit_transform(features)
        scaled_features = pd.DataFrame(mm_features, columns=scaled_features.columns)

    return scaled_features


def create_linear_regression(features: namedtuple, target: namedtuple):
    """
        Create the linear regression model

        Args:
            training_data -> tuple of both training X and Y data
            testing_data -> tuple of both testing X and Y data

        Returns:
            Linear regression model
    """
    reg_model = LinearRegression()
    reg_model.fit(features.training, target.training)
    print(reg_model.score(features.testing, target.testing))
    prediction = reg_model.predict(features.testing)
    print("r^2 and mean square error")
    print(r2_score(target.testing, prediction))
    print(mean_squared_error(target.testing, prediction))
    print("\n")


def main() -> None:
    """
        Main functionality of our linear regression
    """
    # Gather the necessary features
    nba_stats, nba_ws = filter_cols(get_nba_df(from_year=2010))
    nba_pca = apply_pca(nba_stats.fillna(0), dimensions=5)
    std_nba = apply_scaling(nba_stats.fillna(0))
    mm_nba = apply_scaling(nba_stats.fillna(0), scale_type="MinMax")
    std_pca = apply_scaling(nba_pca)
    mm_pca = apply_scaling(nba_pca, scale_type="MinMax")

    # get train testing data
    features, target = get_train_test(nba_stats.fillna(0), nba_ws)
    pca_feats, pca_target = get_train_test(nba_pca, nba_ws)
    std_features, std_target = get_train_test(std_nba, nba_ws)
    mm_features, mm_target = get_train_test(mm_nba, nba_ws)
    std_pca, std_pca_target = get_train_test(std_pca, nba_ws)
    mm_pca, mm_pca_target = get_train_test(mm_pca, nba_ws)

    # Create linear regression models
    create_linear_regression(features, target)
    create_linear_regression(pca_feats, pca_target)
    create_linear_regression(std_features, std_target)
    create_linear_regression(mm_features, mm_target)
    create_linear_regression(std_pca, std_pca_target)
    create_linear_regression(mm_pca, mm_pca_target)


if __name__ == "__main__":
    main()
