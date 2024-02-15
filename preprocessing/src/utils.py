import pandas as pd
import numpy as np
from loguru import logger
import importlib
import os


def _return_values_to_use(selector_dict: dict):

    # Initialize an empty list to store the names of selectors with "usage" set to True
    selected_selectors = []

    # Iterate through the dictionary
    for selector_name, selector_info in selector_dict.items():
        if selector_info["usage"]:
            selected_selectors.append(selector_name)

    return selected_selectors


def _return_cross_validation(local_config: dict):

    cv_class_name = str(list(local_config["cv"].keys())[0])
    cv_params = local_config["cv"][cv_class_name]

    logger.info(f"Loading cross-validation strategy: {cv_class_name}. ")
    cv = getattr(
        importlib.import_module(f"sklearn.model_selection"),
        str(list(local_config["cv"].keys())[0]),
    )(**cv_params)

    return cv


def _plot_cross_validation_scores():

    # https://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_with_cross_validation.html#sphx-glr-auto-examples-feature-selection-plot-rfe-with-cross-validation-py

    tt = 2
