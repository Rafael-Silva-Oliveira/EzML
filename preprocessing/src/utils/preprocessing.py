import pandas as pd
import numpy as np
import seaborn as sns
import json
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import make_column_selector as selector
from sklearn.compose import ColumnTransformer
from loguru import logger


class PreProcessor:

    def __init__(self, config: dict):
        self.config = config

    def main_preprocessor(self, data: pd.DataFrame, dtype: str):
        """_summary_

        Args:
                        data (pd.DatFrame): _description_
                        dtype (string): _description_

        Returns:
                        _type_: _description_
        """
        logger.info(f"Pre-processing {dtype} columns")

        if dtype == "categorical":
            avaliable_preprocessors = {
                "OneHotEncoder": OneHotEncoder(handle_unknown='ignore', sparse_output=False),
                "OrdinalEncoder": OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            }
            settings = self.config["categorical_preprocessing"]
            # Select all columns that are type object (categorical)
            columns_selector = selector(
                dtype_include=object)
            columns = columns_selector(data)

        elif dtype == "numerical":
            avaliable_preprocessors = {
                "StandardScaler": StandardScaler(),
                "MinMaxScaler": MinMaxScaler()}
            settings = self.config["numerical_preprocessing"]
            columns_selector = selector(
                dtype_include=float)
            columns = columns_selector(data)

        data_cp = data.copy()

        # Replace in the categorical preprocessing steps "all" wit the actual columns, excluding the steps that provide specific columns.
        columns_to_exclude = [
            column for sublist_of_columns in [
                col for col in settings.values() if "all" not in col
            ]
            for column in sublist_of_columns
        ]
        columns_excluding_others = [
            col for col in columns if col not in columns_to_exclude]

        for k, sub_columns in settings.items():
            if "" in sub_columns:
                continue
            if "all" in sub_columns:
                settings[k] = columns_excluding_others
            # If all is not in sub_columns, confirm that the specified columns are present in "columns". If not, add them as well. This is necessary since the column selector of type object might not always select the correct columns. Useful if a given column is an integer, but we want to consider it a categorical to apply OneHotEncoder, for example.
            if "all" not in sub_columns:
                columns_missing = [
                    col for col in sub_columns if col not in columns]
                if len(columns_missing) > 0:
                    columns += columns_missing

        # For each specified column, apply the categorical preprocessor
        for col in columns:
            if col not in data.columns:
                logger.error(
                    f"Make sure that the columns you specified are correct or present in the dataframe. {col} column is not present in dataframe columns {data.columns}")
                continue

            try:
                # For each column, grab the associated preprocessor associated with the column to load and pass onto the ColumnTransformer
                preprocessor_name = [
                    preprocessor for preprocessor, columns in settings.items() if col in columns][0]
                preprocessor = avaliable_preprocessors[preprocessor_name]
                logger.info(
                    f"Column {col} will now start being processed using {preprocessor_name}")

            except Exception as e:
                logger.warning(
                    f"The column {col} is considered a {dtype} column and it does not have any preprocessor associated to it. Ignore this warning if this is meant to be the case.")
                continue

            preprocessor = ColumnTransformer(
                [
                    ("transformed", preprocessor,
                     [col]),
                ],
                remainder="drop",
                verbose_feature_names_out=True
            )
            # Create a new dataframe with just the new transformed columns
            new_data = pd.DataFrame(
                preprocessor.fit_transform(data_cp), columns=preprocessor.get_feature_names_out())

            # Grab the index of the current column being transformed in the original dataframe
            col_idx = data_cp.columns.get_loc(col)

            if dtype == "numerical":
                # Insert the re-scalled variable after the original variable
                data_cp.insert(
                    col_idx+1,
                    new_data.columns[0],
                    new_data
                )
                # data_cp[col] = new_data
            elif dtype == "categorical":
                # Keep original and newly transformed columns. This is done differently than with numerical columns since OneHotEncoding gives at least 2 columns which the .insert() method doesn't allow to use. This way, we split the dataframe as before the transformed column, trasnformed column and all the other columns after the transformed column.
                data_cp = pd.concat(
                    [data_cp.iloc[:, :col_idx+1], new_data, data_cp.iloc[:, col_idx+1:]], axis=1)
        return data_cp


def main(CONFIG_PATH: str):
    """_summary_

    Args:
                                                                                                                                                                                                                                                                    CONFIG_PATH (string): _description_
    """
    config = json.load(open(CONFIG_PATH))
    logger.warning("Note: It is important that numerical columns are pre-processed first, before categorical ones. This is to avoid OneHotEncoded columns (binary 0 and 1) to be seen as numerical during numerical pre-processing.")

    data = sns.load_dataset("titanic")

    preprocessor_class = PreProcessor(config=config)

    # Start by pre-processing selected numerical columns
    data = preprocessor_class.main_preprocessor(
        data=data, dtype="numerical")

    # Finally, process categorical columns
    data = preprocessor_class.main_preprocessor(
        data=data, dtype="categorical")

    return data


main(CONFIG_PATH=r"C:\Users\rafaelo\OneDrive - NTNU\Documents\Projects\preprocessing\preprocessing\preprocessing\src\utils\config.json")
