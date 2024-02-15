from train_model import ModelTraining
from preprocessing import PreProcessor
from ead import ExploratoryAnalysis
from loguru import logger
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import json
from datetime import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import json
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    MinMaxScaler,
)
from sklearn.compose import make_column_selector as selector
from sklearn.compose import ColumnTransformer
from loguru import logger
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    RidgeCV,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import importlib
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from datetime import datetime
import os
from pathlib import Path
from sklearn.dummy import DummyClassifier

from sklearn.preprocessing import LabelEncoder

username = os.path.expanduser("~")

date = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")


class Orchestrator(object):

    def __init__(self, config, path_backbone, data_dict):
        self.config = config
        self.path_backbone = path_backbone
        self.data_dict = data_dict

    def run_ExploratoryAnalysis(self) -> dict:
        config = self.config["ead"]
        ExploratoryAnalysisPipeline = ExploratoryAnalysis(config=config)

        result_dictionary = {}

        for command, settings in config.items():
            if settings.get("usage", True):
                match command:
                    case "pandas_profiling_report":
                        p_report = ExploratoryAnalysisPipeline.pandas_profiling(
                            data=self.data_dict["raw_data"]
                        )
                        self.data_dict.setdefault("pandas_report", p_report)
                    case _:
                        # Default case, do nothing or raise an exception
                        pass

    def run_PreProcessor(self) -> dict:
        logger.warning(
            "Note: It is important that numerical columns are pre-processed first, before categorical ones. This is to avoid OneHotEncoded columns (binary 0 and 1) to be seen as numerical during numerical pre-processing."
        )
        config = self.config["preprocessing"]
        PreProcessorPipeline = PreProcessor(config=config)

        data = self.data_dict["raw_data"].copy()

        # Start by pre-processing selected numerical columns
        data = PreProcessorPipeline.encoders(data=data, dtype="numerical")

        # Finally, process categorical columns
        data = PreProcessorPipeline.encoders(data=data, dtype="categorical")

        self.data_dict.setdefault("preprocessed_data", data)

    def run_ModelTraining(self) -> dict:
        config = self.config["modelling"]
        data = self.data_dict["preprocessed_data"].copy()

        ModelTrainingPipeline = ModelTraining(config=config)
        features = config["data"]["features"]
        target = config["data"]["target"]

        # assert config["classification"]["usage"] != config["regression"]["usage"], "You currently have both classification and regression problems set to usage = 1. Please, adjust the settings."

        if config["classification"]["usage"] == 1:
            modelling_problem_type = "classification"

        # elif config["regression"]["usage"] == 1:
        #     modelling_problem_type = "regression"

        model_settings = config[modelling_problem_type]

        # Load full dataset
        data.fillna(0, inplace=True)
        if "all" in features:
            features = data.columns.tolist()
            features.remove(target)
            X = data[features]
        else:
            X = data[features]

        y = pd.DataFrame(data[target])

        if modelling_problem_type == "classification":
            label_encoder = LabelEncoder()
            label_encoder.fit(y)
            y_encoded = label_encoder.transform(y)
            y_decoded = label_encoder.inverse_transform(y_encoded)
            y = pd.DataFrame(y_encoded, columns=[target])

        ModelTrainingPipeline = ModelTraining(config)

        # Split data into trainind, validation and test data
        X_train, X_test, y_train, y_test = ModelTrainingPipeline.tabular_data_split(
            X, y, modelling_problem_type
        )
        # Get best baseline model and optimize it
        if modelling_problem_type == "classification":
            baseline_model = ModelTrainingPipeline.find_best_clf(
                model_settings,
                X_train,
                X_test,
                y_train,
                y_test,
                modelling_problem_type,
                label_encoder,
            )
        elif modelling_problem_type == "regression":
            baseline_model = ModelTrainingPipeline.find_best_reg(
                model_settings, X_train, X_test, y_train, y_test, modelling_problem_type
            )

        # Save best optimized model


def main(CONFIG_PATH: str):
    config = json.load(open(CONFIG_PATH))
    path_backbone = config["path_backbone"]
    data_path = os.path.join(path_backbone, config["data"])
    data = pd.read_csv(data_path)
    directory = path_backbone + "\\" + f"PipelineRun_{date}"

    subdirs = {
        "Model": ["Training Data", "Optimized Model"],
        "Files": [],
        "Plots": [],
    }

    # Check if the directory already exists
    if not os.path.exists(directory):
        for main_dir, sub_dir_list in subdirs.items():
            if sub_dir_list:
                for subdir in sub_dir_list:
                    # TODO: add a flag that adds the model sub-folder if the train model is set to true. Else, don't add that model sub-folder.
                    curr_dir = directory + "\\" + main_dir + "\\" + subdir
                    # Create the directory
                    os.makedirs(curr_dir)
                    print(f"Directory created successfully - {curr_dir=}")

            else:
                curr_dir = directory + "\\" + main_dir
                # Create the directory
                os.makedirs(curr_dir)
                print(f"Directory created successfully - {curr_dir=}")
    else:
        print("Directory already exists or is already populated with files!")

    # Has to be after creating the dir otherwise it will print directory already exists.
    logger.add(f"{directory}\\loguru.log")
    logger.info(f"Directory where outputs will be saved: {directory}")

    data_dict = {"raw_data": data}
    ORCHESTRATOR = Orchestrator(
        config=config, path_backbone=path_backbone, data_dict=data_dict
    )
    ORCHESTRATOR.run_ExploratoryAnalysis()
    ORCHESTRATOR.run_PreProcessor()
    ORCHESTRATOR.run_ModelTraining()


if __name__ == "__main__":
    main(
        CONFIG_PATH=r"C:\Users\rafaelo\OneDrive - NTNU\Documents\Projects\preprocessing\preprocessing\preprocessing\src\config.json"
    )
