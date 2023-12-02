# SyntheticTrialData
This repository provides a codebase for the generation of static and longitudinal synthetic clinical trial data. This code was developed and published as part of a master thesis at the TU Berlin.

## Installation Guide
#### Install Dependencies with Poetry
1. Clone Repository
2. Make sure you installed poetry:
    ```bash
    curl -sSL https://install.python-poetry.org | python3 -
    ```
3. In your project directory, run the following to install required dependecies:
    ```bash
    poetry lock
    poetry install
    ```
## Generate static synthetic clinical trial data
1. Define data schema according to `config/dataset/data_schema.yaml`
2. Configure settings of the experiment in `config/config_static_only.yaml`
3. For the generation of **static** synthetic trial data run
    ```bash
    poetry run python train_static_only.py
    ```

## Generate longitudinal synthetic clinical trial data
1. Define data schema according to `config/dataset/data_schema.yaml`
2. Configure settings of the experiment in `config/config.yaml`
3. For the generation of **longitudinal** synthetic trial data run
    ```bash
    poetry run python train.py
    ```


