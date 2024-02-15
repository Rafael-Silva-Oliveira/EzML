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
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    RidgeCV,
)
from sklearn.naive_bayes import (
    GaussianNB,
    CategoricalNB,
    MultinomialNB,
    BernoulliNB,
    ComplementNB,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_selection import (
    SequentialFeatureSelector,
    SelectFromModel,
    SelectKBest,
    f_classif,
    RFECV,
)
import importlib
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from datetime import datetime
import os
from pathlib import Path
from sklearn.dummy import DummyClassifier

from sklearn.preprocessing import LabelEncoder

from utils import _return_values_to_use, _return_cross_validation


class ModelTraining:
    def __init__(self, config):
        self.config = config

    def tabular_data_split(self, X, y, modelling_problem_type):
        logger.info(f"Creating data splits for model training.")

        # Select settings for each modelling problem
        if modelling_problem_type == "classification":
            val, test_size, shuffle, stratify = [
                self.config[modelling_problem_type]["train_test_split"][
                    "validation_size"
                ],
                self.config[modelling_problem_type]["train_test_split"]["test_size"],
                self.config[modelling_problem_type]["train_test_split"]["shuffle"],
                self.config[modelling_problem_type]["train_test_split"]["stratify"],
            ]

            logger.warning(
                f"Shuffle is currently set to {shuffle}. If you're using time-series data, this is not advisable."
            )

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=0, shuffle=shuffle
            )
            # X_val, X_test, y_val, y_test = train_test_split(
            #     X_hold_out, y_hold_out, test_size=test, shuffle=shuffle, random_state=0)

        if modelling_problem_type == "regression":
            val_size, test_size, shuffle, stratify = [
                self.config[modelling_problem_type]["train_test_split"][
                    "validation_size"
                ],
                self.config[modelling_problem_type]["train_test_split"]["test_size"],
                self.config[modelling_problem_type]["train_test_split"]["shuffle"],
                self.config[modelling_problem_type]["train_test_split"]["stratify"],
            ]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=0, shuffle=shuffle
            )
            # X_val, X_test, y_val, y_test = train_test_split(
            #     X_hold_out, y_hold_out, test_size=test, shuffle=shuffle, random_state=0)

        logger.info(
            f"\nShape of training data: {X_train.shape}. \nShape of test data: {X_test.shape}\nUsing a split with a validation size of {val} out of {1-val} training split - {val*(1-val)} is the real hold-out/validation set size.\nThe actual test size is {test_size} (1 -  training split)."
        )
        return X_train, X_test, y_train, y_test

    def find_best_clf(
        self,
        model_settings,
        X_train,
        X_test,
        y_train,
        y_test,
        modelling_problem_type,
        label_encoder,
    ):
        logger.info(f"Finding best baseline classifier.")

        best_baseline_model = None
        param_distribution = None
        best_baseline_model_cv_results = None
        models = model_settings["models"]

        # Load dummy classifier with most frequent categorization and use the score as baseline
        strategy = "most_frequent"
        dummy_clf = DummyClassifier(strategy=strategy)
        dummy_clf.fit(X_train, y_train)
        dummy_clf.predict(X_train)
        # Accuracy is the score being used! #TODO: add possibility of changing the metric being used
        best_score = dummy_clf.score(X_train, y_train)

        logger.info(
            f"DummyClassifier with {strategy} was used to compute the baseline score for baseline model search. The DummyClassifier score is: {best_score}"
        )

        # Transforming y data to a numpy array to avoid unnecessary warnings.
        y_train = y_train.T.to_numpy()[0]
        y_test = y_test.T.to_numpy()[0]

        for model_type, model_config in models.items():
            for model_name, model_utils in model_config.items():
                if model_utils["usage"] == 1:
                    logger.info(f"Running {model_name}. ")

                    model = getattr(
                        importlib.import_module(f"sklearn.{model_type}"), model_name
                    )()

                    if model_utils["run_feature_selection"]:

                        # Run feature selection TODO: see a way to add the feature selector to the pipeline
                        selectors = _return_values_to_use(
                            self.config["feature_selection"]
                        )

                        assert (
                            len(selectors) == 1
                        ), f"Feature selection methods can only be 1 but you chose {len(selectors)}. Current methods selected are {selectors = }. Please, change it to only have one method with the setting 'usage' set to True."

                        logger.info(
                            f"Running feature selection with {selectors[0]} for {model_name}. "
                        )

                        selector, selected_features = self.feature_selection(
                            selectors[0],
                            model,
                            self.config,
                            X_train,
                            y_train,
                        )
                        X_train, X_test = (
                            X_train[selected_features],
                            X_test[selected_features],
                        )

                    pipeline = make_pipeline(model)

                    hyperparameter_settings = model_utils["hyperparameters"][
                        "hyperparameter_settings"
                    ]

                    logger.info(f"Running CV for {model_name}.")

                    # Load cross validation strategy dynamically
                    cv = _return_cross_validation(hyperparameter_settings)

                    logger.info(
                        f"Running cross-validation to find best baseline model. "
                    )

                    # Apply cross validation to get the best baseline model to be then further optimized
                    cv_results = cross_validate(
                        pipeline,
                        X_train,
                        y_train,
                        cv=cv,
                        scoring=hyperparameter_settings["scoring"],
                        return_estimator=True,
                        return_train_score=True,
                        error_score="raise",
                    )

                    # This "test_scoring_type" is actually the validation set. The actual test set will only be used to calculate the final classification report to avoid data leakage.Do note that if more scorings are added to the list, only the first one will be used to evaluate the best model score

                    eval_score = hyperparameter_settings["scoring"][0]
                    logger.info(
                        f"Score being used to calculate the best validation score is: {eval_score}"
                    )

                    current_model_score = cv_results[f"test_{eval_score}"].mean()

                    if current_model_score > best_score:
                        best_score = current_model_score
                        best_baseline_model = pipeline
                        param_distribution = model_utils["hyperparameters"][
                            "param_distribution"
                        ]
                        best_baseline_model_cv_results = cv_results
        # Optimizing the best baseline model on training data
        logger.info(
            f"Optimizing best baseline model using RandomizedSearchCV. \n Best baseline model is: {best_baseline_model}"
        )

        # Transform strings in the param_distribution to actual ranges
        param_distribution_final = {
            key: (
                eval(value)
                if isinstance(value, str) and value.startswith("np.")
                else value
            )
            for key, value in param_distribution.items()
        }

        randomized_search = RandomizedSearchCV(
            best_baseline_model,  # Choose just the model and not the imputer
            param_distributions=param_distribution_final,
            n_iter=hyperparameter_settings["n_iter"],
            scoring=hyperparameter_settings["scoring"],
            cv=cv,
            random_state=42,
            n_jobs=hyperparameter_settings["n_jobs"],
            verbose=3,
            refit=hyperparameter_settings["scoring"][0],
        ).fit(X_train, y_train)

        # Get information for the best optimized model
        best_model = randomized_search.best_estimator_
        best_model_name = list(best_model.named_steps.keys())[0]

        # Get test score (no need for this as its already in the classifiction report down below)
        test_score = best_model.score(X_test, y_test)

        # Get the predicted using test data from the best optimized model
        y_pred = best_model.predict(X_test)

        # Decode the test and predicted for output metric file
        y_test_decoded = label_encoder.inverse_transform(y_test)
        y_pred_decoded = label_encoder.inverse_transform(y_pred)

        logger.info("Calculating classification report on test dataset.")

        # Get classification report on test data (avoids data leakage)
        try:
            metrics_results = classification_report(
                y_pred=y_pred_decoded, y_true=y_test_decoded, output_dict=True
            )
        except Exception as e:
            print(
                f"When calculationg the predictions this exception occured:/n{e}./n/nMake sure that you manually tagged the data otherwise NaNs will appear on the true labels."
            )

        # Create metrics structure for the best model
        models_metrics = {
            modelling_problem_type: {best_model: metrics_results}
        }  # create a dictionary with current modelling_problem_type and inside this key, create a new dictionary as the value with the current model being passed and the metric results

        results = pd.json_normalize(models_metrics)
        # flatten the dicts but then split all columns on `.` into different column levels
        results.columns = pd.MultiIndex.from_tuples(
            [tuple(col.split(".")) for col in results.columns]
        )
        results = results.T  # transpose the data so its more readable

        date = datetime.now().strftime("%Y_%m_%d_%I_%M_%S_%p")

        new_path = r"{}/OneDrive - NTNU/Documents/Projects/preprocessing/preprocessing/preprocessing/data/processed/metrics_results/{}".format(
            Path.home(), modelling_problem_type
        )
        isExist = os.path.exists(new_path)
        if not isExist:
            # Create a new directory because it/ does not exist
            os.makedirs(new_path)
            print(
                f"The new directory is created. You can now check the metrics_results for {best_model_name} in this directory {new_path}"
            )
        results.to_excel(
            r"{}/{}_{}_{}.xlsx".format(
                new_path, modelling_problem_type, best_model_name, date
            )
        )

    # TODO: add validation curve to check for overfitting
    # TODO: save best model as JSON

    def find_best_reg(self, model_settings, X_train, X_test, y_train, y_test):
        logger.info(f"Creating data splits for model training.")

        ...

    def nested_cross_validation(self):
        logger.info(f"Creating data splits for model training.")

        ...

    def ensemble_pipeline(self): ...

    def generalization_clf_metrics(self): ...

    def generalization_reg_metrics(self): ...

    def feature_selection(self, selector_type, model, config, X_train, y_train):

        logger.info(f"Running feature selection for {model} with {selector_type}")
        local_config = config["feature_selection"][selector_type]["params"]

        if selector_type == "SequentialFeatureSelector":

            cv = _return_cross_validation(local_config)
            selector = SequentialFeatureSelector(
                model,
                n_features_to_select=local_config["n_features_to_select"],
                direction=local_config["direction"],
                scoring=local_config["scoring"],
                cv=cv,
            )

            selector_fitted = selector.fit(X_train, y_train)
            selected_features = [
                col for col in selector_fitted.get_feature_names_out(data.columns)
            ]

        elif selector_type == "RFECV":
            cv = _return_cross_validation(local_config)
            selector = RFECV(
                model,
                min_features_to_select=local_config["min_features_to_select"],
                step=local_config["step"],
                scoring=local_config["scoring"],
                cv=cv,
                verbose=local_config["verbose"],
                n_jobs=local_config["n_jobs"],
                importance_getter=local_config["importance_getter"],
            )

            selector_fitted = selector.fit(X_train, y_train)
            selected_features = [
                col for col in selector_fitted.get_feature_names_out(X_train.columns)
            ]

        return selector_fitted, selected_features

    def feature_engineering(self): ...
