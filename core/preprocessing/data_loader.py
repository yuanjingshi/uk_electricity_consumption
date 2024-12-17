import json
import os
from pathlib import Path

import click
import kaggle


def kaggle_api_key(user_name: str | None = None, api_key: str | None = None) -> None:
    """
    Function that takes the kaggle API key as input and saves it to
    ~/.kaggle/kaggle.json if it doesn't exist
    """
    if user_name is not None and api_key is not None:
        # Create ~/.kaggle directory if it doesn't exist
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_dir.mkdir(parents=True, exist_ok=True)

        # Path to kaggle.json
        kaggle_json = kaggle_dir / "kaggle.json"

        # Create the JSON content
        api_config = {"username": user_name, "key": api_key}

        # Write the config file if it doesn't exist
        if not kaggle_json.exists():
            with open(kaggle_json, "w") as f:
                json.dump(api_config, f)

            # Set file permissions to be readable only by the user
            os.chmod(kaggle_json, 0o600)
            print(f"Kaggle API key has been saved to {kaggle_json}")


def kaggle_download_data(dataset: str, save_path: str | None = None) -> None:
    """
    Function that downloads the NBA database from Kaggle
    """
    # Create the data directory if it doesn't exist
    root_path = Path(__file__).parent.parent
    if save_path is not None:
        data_path = root_path / save_path
    else:
        data_path = root_path / "data"
    data_path.mkdir(parents=True, exist_ok=True)

    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(dataset, path=data_path, unzip=True)
    print(f"Data has been downloaded to {data_path}")


@click.command()  # type: ignore
@click.option(  # type: ignore
    "--dataset",
    "-d",
    required=True,
    help='Kaggle dataset identifier (e.g., "username/dataset-name")',
)
@click.option("--save-path", "-p", required=True, help="Data save path")  # type: ignore
@click.option(  # type: ignore
    "--user-name",
    "-u",
    required=False,
    help="Your Kaggle username",
)
@click.option(
    "--api-key", "-k", required=False, help="Your Kaggle API key"
)  # type: ignore
def main(
    dataset: str,
    save_path: str | None = None,
    user_name: str | None = None,
    api_key: str | None = None,
) -> None:
    kaggle_api_key(user_name=user_name, api_key=api_key)
    kaggle_download_data(dataset=dataset, save_path=save_path)


if __name__ == "__main__":
    main()
