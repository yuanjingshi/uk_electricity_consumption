import pandas as pd

from core.feature_engineering.feature_creation import create_features


def test_create_features():
    # Create a DataFrame with dates
    date_range = pd.date_range(start="2022-01-01", end="2022-12-31")
    df = pd.DataFrame(date_range, columns=["date"])
    df.set_index("date", inplace=True)
    df_result = create_features(df)

    # Randomly sample the dates in date_range
    sampled_dates = df.sample(n=5, random_state=1)

    # Get the day of the week and week of the year for the sampled dates
    sampled_dates["day_of_week"] = sampled_dates.index.day_of_week
    sampled_dates["week_of_year"] = sampled_dates.index.isocalendar().week.astype(
        "int64"
    )
    sampled_dates["month"] = sampled_dates.index.month
    sampled_dates["quarter"] = sampled_dates.index.quarter
    sampled_dates["year"] = sampled_dates.index.year

    for d in sampled_dates.index:
        expected = sampled_dates.loc[d]
        actual = df_result.loc[d]
        pd.testing.assert_series_equal(expected, actual, check_dtype=True)
