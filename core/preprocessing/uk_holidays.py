import logging
from datetime import date
from typing import List

import holidays


# Get UK holidays with observed holidays
def retrieve_holiday_dates(
    subdiv: str = "England",
    from_date: date = date(2009, 1, 1),
    to_date: date = date(2009, 12, 31),
) -> dict:
    from_year = int(from_date.year)
    to_year = int(to_date.year) + 1

    bank_holiday = holidays.UK(
        subdiv=subdiv, years=range(from_year, to_year), observed=True
    ).items()

    holiday_dates_observed: List
    holiday_dates_observed_name: List

    for d, name in sorted(bank_holiday):
        # Pop the previous value as observed bank holidays takes place later
        if "observed" in name:
            logging.info(f"Observed Holiday: {d} and {name}")
            if len(holiday_dates_observed) > 0:
                holiday_dates_observed.pop()
                holiday_dates_observed_name.pop()

        if from_date <= d <= to_date:
            d = d.strftime("%Y-%m-%d")
            holiday_dates_observed.append(d)
            holiday_dates_observed_name.append(name)
        else:
            break

    holiday_dict = dict(zip(holiday_dates_observed, holiday_dates_observed_name))
    return holiday_dict
