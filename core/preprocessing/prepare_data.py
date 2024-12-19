import pandas as pd


def fill_zero_consumptions(df):
    """
    Fills zero consumption values in the DataFrame with the
    mean consumption for the corresponding year and settlement period.

    Parameters:
    df (pandas.DataFrame): DataFrame containing electricity consumption data
    with columns 'settlement_date', 'settlement_period', and 'tsd'.

    Returns:
    pandas.DataFrame: DataFrame with zero consumption values
    filled with the mean consumption for the corresponding
    year and settlement period.
    """

    df["year"] = pd.to_datetime(df["settlement_date"]).dt.year
    mean_consumption_per_period = df.groupby(["year", "settlement_period"])[
        "tsd"
    ].mean()

    zero_consumptions = df.loc[df["tsd"] == 0.0, ["year", "settlement_period"]]
    for index, row in zero_consumptions.iterrows():
        year = row["year"]
        period = row["settlement_period"]
        mean_tsd = mean_consumption_per_period.loc[year, period]
        df.loc[index, "tsd"] = int(mean_tsd)

    return df


def fill_missing_weather_data(df):
    """
    Fills missing weather data in the DataFrame with
    the mean value for the corresponding month.

    Parameters:
    df_weather_by_date (pandas.DataFrame):
    DataFrame containing weather data with a DateTime index.

    Returns:
    pandas.DataFrame:
    DataFrame with missing weather data filled
    with the mean value for the corresponding month.
    """
    for c in df.columns:
        df[c] = df.groupby(df.index.month)[c].transform(
            lambda x: x.fillna(x.mean())
        )
    return df
