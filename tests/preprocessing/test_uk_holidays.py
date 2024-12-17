import datetime as dt

import pytest

from core.preprocessing.uk_holidays import retrieve_holiday_dates


@pytest.mark.parametrize(
    "subdiv, from_date, to_date",
    [
        ("England", dt.date(2022, 1, 1), dt.date(2023, 12, 31)),
        ("Wales", dt.date(2022, 1, 1), dt.date(2023, 12, 31)),
        ("Scotland", dt.date(2022, 1, 1), dt.date(2023, 12, 31)),
        ("Northern Ireland", dt.date(2022, 1, 1), dt.date(2023, 12, 31)),
    ],
)
def test_uk_holidays(subdiv, from_date, to_date):
    loaded_holidays = retrieve_holiday_dates(subdiv, from_date, to_date)
    assert (
        "2023-12-25" in loaded_holidays.keys()
    ), "Christmas Day should be a holiday"
    assert (
        "2023-05-08" in loaded_holidays.keys()
    ), "King's Coronation Day should be a holiday"
    assert (
        "2022-06-02" in loaded_holidays.keys()
    ), "Queen's Jubilee should be a holiday"
    assert (
        "2023-01-01" not in loaded_holidays.keys()
    ), "A New Year's day that falls on the weekend \
            will be substituted by the next working day"
    if subdiv == "Scotland":
        assert (
            "2023-01-03" in loaded_holidays.keys()
        ), f"A substitute day for New Year's day should be a holiday in {subdiv}"
    else:
        assert (
            "2023-01-02" in loaded_holidays.keys()
        ), f"A substitute day for New Year's day should be a holiday in {subdiv}"
