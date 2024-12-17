# uk_electricity_consumption

This is a final project for data science

## Environment Setup

Install the virtual environment and install your repo from source:

```bash
conda env create -f environment.yml
conda activate uk_demand
pip install --no-build-isolation -e .
```

To install the pre-commit hooks, run:

```bash
pre-commit install
```

When updating the pre-commit hooks, run:

```bash
pre-commit clean
pre-commit install
```

To run pre-commit hooks on all files, run:

```bash
pre-commit run --all-files
```

to run it on a specific file, run:

```bash
pre-commit run --files <file>
```

## Getting the data

1. create kaggle account: https://www.kaggle.com/
2. get API keys and user name
   - Go to your account, and click on your profile in the top right
   - Then click on "Settings"
   - Scroll down to the API section and click on "Create New Token"
   - Get your username and API key from there

Then run the following command in the terminal being at the root of the repository. Replace
"your_user_name" and "your_api_key" with your username and API key. This creates a json file at
"~/.kaggle/kaggle.json" with your username and API key. This is used to authenticate your account
and download the data.

```bash
python core/data_load/data_loader.py -u "your_user_name" -k "your_api_key" -d "albertovidalrod/electricity-consumption-uk-20092022" -p "data/demand_data"
python core/data_load/data_loader.py -u "your_user_name" -k "your_api_key" -d "jakewright/2m-daily-weather-history-uk" -p "data/weather_data"
```
