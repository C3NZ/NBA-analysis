import pandas as pd

from clean import get_nba_df


def filter_cols(df: pd.DataFrame) -> tuple:
    # Columns we want to remove
    unwanted_cols = ["Year", "Pos", "Age", "Tm", "blanl", "blank2", "OWS", "DWS"]

    # Target column we'd like
    target_col = ["WS"]
    nba_stats = df.drop(columns=unwanted_cols)
    nba_ws = df[target_col]

    return nba_stats, nba_ws


def main():
    nba_stats, nba_ws = filter_cols(get_nba_df())
    print(nba_stats)
    print(nba_ws)


if __name__ == "__main__":
    main()
