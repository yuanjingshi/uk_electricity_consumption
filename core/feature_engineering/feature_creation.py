def create_features(df):
    """
    Create time-based features from the datetime index of the DataFrame.

    Parameters:
    df (pandas.DataFrame): Input DataFrame with a datetime index.

    Returns:
    pandas.DataFrame: DataFrame with additional columns for day of the week,
                      week of the year, month, quarter, and year.
    """

    df = df.copy()
    df["day_of_week"] = df.index.day_of_week
    df["week_of_year"] = df.index.isocalendar().week.astype("int64")
    df["month"] = df.index.month
    df["quarter"] = df.index.quarter
    df["year"] = df.index.year

    return df
