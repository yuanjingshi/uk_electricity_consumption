import numpy as np
import pandas as pd


def create_sample_split(df: pd.DataFrame, threshold_date: str):
    """
    Splits the DataFrame into training and testing samples based on a threshold date.
    Parameters:
    df (pd.DataFrame): The input DataFrame with a datetime index.
    threshold_date (str):
    The date in string format to split the DataFrame. Rows with an index
    earlier than this date will be assigned to the training sample,
    and rows with an index on or after this date will be assigned to the testing sample.
    Returns:
    pd.DataFrame:
    The DataFrame with an additional column 'sample' indicating whether each row
    belongs to the 'train' or 'test' sample.
    """

    threshold_date = pd.to_datetime(threshold_date)
    df["sample"] = np.where(df.index < threshold_date, "train", "test")
    return df
