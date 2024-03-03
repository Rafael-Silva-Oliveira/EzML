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
    SGDClassifier,
    RidgeClassifier,
    Perceptron,
)
from sklearn.neighbors import KNeighborsClassifier
import xgboost
import catboost
import lightgbm
import optuna

# from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

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
    AdaBoostClassifier,
)

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_selection import (
    SequentialFeatureSelector,
    SelectFromModel,
    SelectKBest,
    f_classif,
    RFECV,
)

from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt

import importlib
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_curve, auc
from datetime import datetime
import os
from pathlib import Path
from sklearn.dummy import DummyClassifier

from sklearn.preprocessing import LabelEncoder
from joblib import dump, load

from utils import _return_values_to_use, _return_cross_validation
import copy

sns.set(font_scale=1.5)
color = "Spectral"
color_hist = "teal"
sns.set_style("ticks")


class ModelTraining:
    def __init__(self, config, saving_path):
        self.config = config
        self.saving_path = saving_path

    def tabular_data_split(self, X, y, modelling_problem_type):
        logger.info(f"Creating data splits for model training.")
        logger.info(f"Data shape: {X.shape}")
        logger.info(f"Label distribution:\n{y.value_counts()}")

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
            if stratify:
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=test_size,
                    random_state=0,
                    shuffle=shuffle,
                    stratify=y,
                )
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=test_size,
                    random_state=0,
                    shuffle=shuffle,
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

            if stratify:
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=test_size,
                    random_state=0,
                    shuffle=shuffle,
                    stratify=y,
                )
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=test_size,
                    random_state=0,
                    shuffle=shuffle,
                )
            # X_val, X_test, y_val, y_test = train_test_split(
            #     X_hold_out, y_hold_out, test_size=test, shuffle=shuffle, random_state=0)

        logger.info(
            f"\nShape of training data: {X_train.shape}. \nShape of test data: {X_test.shape}\nUsing a split with a validation size of {val} out of {1-val} training split - {val*(1-val)} is the real hold-out/validation set size.\nThe actual test size is {test_size} (1 -  training split)."
        )
        return X_train, X_test, y_train, y_test

    def train_and_optimize_clf(
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
        cv_results_dict = {}
        training_plotting_metrics = {}
        test_plotting_metrics = {}
        models = model_settings["models"]
        best_model_selected_features = None

        # Load dummy classifier with most frequent categorization and use the score as baseline
        dummy_clf = DummyClassifier(
            strategy=model_settings["DummyClassifier"]["strategy"]
        )
        dummy_clf.fit(X_train, y_train)
        dummy_clf.predict(X_train)

        # NOTE: Accuracy is the score being used!
        best_score = dummy_clf.score(X_train, y_train)
        dummy_score = dummy_clf.score(X_train, y_train)

        logger.info(
            f"DummyClassifier with {model_settings['DummyClassifier']['strategy']} was used to compute the baseline score for baseline model search. The DummyClassifier score is: {best_score}. NOTE: The score for the dummy classifier is 'accuracy'."
        )

        # Transforming y data to a numpy array to avoid unnecessary warnings.
        y_train = y_train.T.to_numpy()[0]
        y_test = y_test.T.to_numpy()[0]

        # Gather CV settings
        cv_settings = model_settings["cv_settings"]
        eval_score = cv_settings["scoring"][0]

        # Load cross validation strategy dynamically
        cv = _return_cross_validation(cv_settings)

        # Get optimization type
        optimization = model_settings["optimization"]

        optimization_type_list = [
            key
            for key, value in model_settings["optimization"].items()
            if value["usage"]
        ]

        if len(optimization_type_list) > 1:
            raise ValueError(
                f"Only one optimization type can be used. You currently have {len(optimization_type_list)} types selected. Please, adjust the settings."
            )

        optimization_type = optimization_type_list[0]

        for model_type, model_config in models.items():
            for model_name, model_utils in model_config.items():
                if model_utils["usage"] == 1:
                    logger.info(f"Running {model_name}. ")

                    X_train_cp = copy.deepcopy(X_train)
                    X_test_cp = copy.deepcopy(X_test)
                    y_train_cp = copy.deepcopy(y_train)

                    if "XGB" in model_name:
                        model = eval(f"xgboost.{model_name}()")
                    elif "CatBoost" in model_name:
                        model = eval(f"catboost.{model_name}()")
                    elif "LGBMClassifier" in model_name:
                        model = eval(f"lightgbm.{model_name}()")
                    else:
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
                            X_train_cp,
                            y_train_cp,
                        )
                        X_train_cp, X_test_cp = (
                            X_train_cp[selected_features],
                            X_test_cp[selected_features],
                        )
                        logger.info(
                            f"Length of selected features for {model_name} is {len(selected_features)}"
                        )

                    pipeline = make_pipeline(model)

                    # Transform strings in the param_distribution to actual ranges
                    param_distribution_final = {
                        key: (
                            eval(value)
                            if isinstance(value, str) and value.startswith("np.")
                            else value
                        )
                        for key, value in model_utils["hyperparameters"][
                            "param_distribution"
                        ].items()
                    }

                    # Optimizing the best baseline model on training data
                    logger.info(
                        f"Optimizing baseline model using '{optimization_type}'. \n Baseline model is {model_name} and it will be optimized with the following parameters: \n{param_distribution_final}"
                    )

                    if optimization_type == "optuna":
                        study = self.optimize_clf(
                            param_distribution_final,
                            model,
                            X_train_cp,
                            y_train_cp,
                            cv,
                            cv_settings,
                        )

                        best_trial = study.best_trial
                        best_params = best_trial.params
                        best_cv_sore = best_trial.value

                        print(f"Best hyperparameters for {model_name}: {best_params}")
                        print(f"Best score: {best_cv_sore}")

                        optimized_clf = model.set_params(**best_params)

                        optimized_clf.fit(X_train_cp, y_train_cp)

                    elif optimization_type == "RandomizedSearchCV":
                        # Optimize hyperparameters
                        optimized_clf = RandomizedSearchCV(
                            model,  # Choose just the model and not the imputer
                            param_distributions=param_distribution_final,
                            n_iter=cv_settings["n_iter"],
                            scoring=cv_settings["scoring"],
                            cv=cv,
                            random_state=0,
                            n_jobs=cv_settings["n_jobs"],
                            verbose=3,
                            refit=cv_settings["scoring"][0],
                            return_train_score=True,
                        ).fit(X_train_cp, y_train_cp)

                    # Apply cross validation to get the best model
                    if hasattr(optimized_clf, "best_estimator_"):
                        cv_results = cross_validate(
                            optimized_clf.best_estimator_,
                            X_train_cp,
                            y_train_cp,
                            cv=cv,
                            scoring=cv_settings["scoring"],
                            return_estimator=True,
                            return_train_score=True,
                            error_score="raise",
                        )
                    else:
                        cv_results = cross_validate(
                            optimized_clf,
                            X_train_cp,
                            y_train_cp,
                            cv=cv,
                            scoring=cv_settings["scoring"],
                            return_estimator=True,
                            return_train_score=True,
                            error_score="raise",
                        )

                    cv_results_dict[model_name] = cv_results
                    # This "test_scoring_type" is actually the validation set. The actual test set will only be used to calculate the final classification report to avoid data leakage.Do note that if more scorings are added to the list, only the first one will be used to evaluate the best model score

                    cv_results_test = cv_results[f"test_{eval_score}"]
                    cv_results_train = cv_results[f"train_{eval_score}"]
                    current_model_score = cv_results_test.mean()

                    training_plotting_metrics[model_name] = {
                        f"test_{eval_score}": current_model_score,
                        "std": cv_results_test.std(),
                    }

                    # NOTE: it doesn't make sense to add the Confusion Matrix for the traning data since the model has already been trained and seen the training data. If we fit the model again on the training data, it will provide much better results in the confusion matrix than the results that were seen in the training and validation scores. Confusion Matrix is only used on the test data
                    self.plot_training_curves(
                        optimized_clf,
                        eval_score=eval_score,
                        test_label_name="val",
                        model_name=model_name,
                        cv_results=cv_results,
                        use_from_cross_val=False,
                    )

                    # Use the optimized model to predict on the actual test set (unseen data)
                    test_results, pred_metrics_dict, y_pred_decoded = self.predict_clf(
                        optimized_clf,
                        model_settings,
                        X_test_cp,
                        y_test,
                        modelling_problem_type,
                        label_encoder,
                    )

                    # Save metrics and predictions for ROC curve on unseen data
                    test_plotting_metrics[model_name] = {
                        f"test_{eval_score}": pred_metrics_dict[modelling_problem_type][
                            model_name
                        ][eval_score],
                        "preds": y_pred_decoded,
                    }

                    test_model_score = pred_metrics_dict[modelling_problem_type][
                        model_name
                    ][eval_score]

                    logger.info(
                        f"\n\n- Score being used to attain the best model is '{eval_score}'.\n\n- The scores for the optimized version with '{optimization_type}' of the '{model_name}' model are:\n\n- Validation score:\n"
                        f"{cv_results_test.mean():.3f} ± {cv_results_test.std():.3f}.\n\n- Training score:\n"
                        f"{cv_results_train.mean():.3f} ± {cv_results_train.std():.3f}. \n\n- Test score:\n{test_model_score:.3f}"
                    )

                    # Check if test score is better than the previous best score. If so, save the current parameters and results as well as the best classifier (best classifier = the classifier that performed best on the unseen test data)
                    if test_model_score > best_score:
                        best_score = test_model_score
                        best_baseline_model = optimized_clf
                        param_distribution = model_utils["hyperparameters"][
                            "param_distribution"
                        ]
                        best_baseline_model_cv_results = cv_results
                        best_model_test_results = pred_metrics_dict
                        best_model_X_test = X_test_cp

        if hasattr(best_baseline_model, "best_estimator_"):
            logger.info(
                f"Best model is: {best_baseline_model.best_estimator_.__class__.__name__} and it has the following optimized parameters:\n\n {best_baseline_model.best_params_}\n\n... with a test '{eval_score}' of {best_score:.3f}"
            )
        else:
            logger.info(
                f"Best model is: {best_baseline_model.__class__.__name__} and it has the following optimized parameters:\n\n {best_baseline_model.get_params}\n\n... with a test '{eval_score}' of {best_score:.3f}"
            )

        # Plot the training metrics
        self.plot_CV_results(
            eval_score,
            training_plotting_metrics,
            self.saving_path,
            prefix="Training/Validation",
        )
        self.plot_clfs(
            eval_score,
            training_plotting_metrics,
            test_plotting_metrics,
            y_test,
            self.saving_path,
            dummy_score,
        )

        return (
            best_baseline_model,
            cv_results_dict,
            param_distribution,
            label_encoder,
            best_model_X_test,
        )

    # Define the objective function

    def optimize_clf(
        self,
        param_distribution,
        model,
        X_train_cp,
        y_train_cp,
        cv,
        cv_settings,
    ):
        from sklearn.model_selection import cross_val_score

        def objective(
            trial,
            param_distribution,
            model,
            X_train_cp,
            y_train_cp,
            cv,
            cv_settings,
        ):
            from sklearn.metrics import accuracy_score

            params = {
                k: trial.suggest_categorical(k, v)
                for k, v in param_distribution.items()
            }
            model.set_params(**params)
            return cross_val_score(
                model,
                X_train_cp,
                y_train_cp,
                cv=cv,
                scoring=cv_settings["scoring"][0],
            ).mean()

            return cross_val_score

        # Create a study object and optimize it
        study = optuna.create_study(direction="maximize")

        study.optimize(
            lambda trial: objective(
                trial=trial,
                param_distribution=param_distribution,
                model=model,
                X_train_cp=X_train_cp,
                y_train_cp=y_train_cp,
                cv=cv,
                cv_settings=cv_settings,
            ),
            n_trials=25,
        )

        return study

    def predict_clf(
        self,
        best_baseline_model,
        model_settings,
        X_test,
        y_test,
        modelling_problem_type,
        label_encoder,
        prefix="Test",
    ):

        # Get information for the best optimized model
        if hasattr(best_baseline_model, "best_estimator_"):
            best_model = best_baseline_model.best_estimator_
        else:
            best_model = best_baseline_model
        best_model_name = best_model.__class__.__name__

        # Get test score (no need for this as its already in the classifiction report down below)
        test_score = best_model.score(X_test, y_test)

        # Get the predicted using test data from the best optimized model
        y_pred = best_model.predict(X_test)

        # Decode the test and predicted for output metric file
        y_test_decoded = label_encoder.inverse_transform(y_test)
        y_pred_decoded = label_encoder.inverse_transform(y_pred)

        logger.info(
            f"Calculating classification report on the unseen test dataset with the best optimized model: {best_model_name}."
        )

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
            modelling_problem_type: {best_model_name: metrics_results}
        }  # create a dictionary with current modelling_problem_type and inside this key, create a new dictionary as the value with the current model being passed and the metric results

        results = pd.json_normalize(models_metrics)
        # flatten the dicts but then split all columns on `.` into different column levels
        results.columns = pd.MultiIndex.from_tuples(
            [tuple(col.split(".")) for col in results.columns]
        )
        results = results.T  # transpose the data so its more readable

        date = datetime.now().strftime("%Y_%m_%d_%I_%M_%S_%p")

        results.to_excel(
            r"{}/Files/{}_{}_{}_{}.xlsx".format(
                self.saving_path, prefix, modelling_problem_type, best_model_name, date
            )
        )
        self.plot_conf_matrix(
            best_model, X_test, y_test, model_name=best_model_name, prefix=prefix
        )

        return results, models_metrics, y_pred_decoded

    def save_best_model(self, best_model):

        logger.info(f"Saving the best model:\n\n{best_model}")
        if hasattr(best_model, "best_estimator_"):
            model = best_model.best_estimator_
        else:
            model = best_model

        best_model_name = model.__class__.__name__
        dump(
            model,
            f"{self.saving_path}/Model/Best Optimized Model/{best_model_name}.joblib",
        )

    def plot_conf_matrix(self, clf, X, y, model_name, prefix):
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

        if hasattr(clf, "best_estimator_"):
            disp = ConfusionMatrixDisplay.from_estimator(clf.best_estimator_, X, y)
        else:
            disp = ConfusionMatrixDisplay.from_estimator(clf, X, y)

        disp.ax_.set_title(f"{prefix} Confusion Matrix for {model_name}")
        # plt.show()
        with plt.rc_context():  # Use this to set figure params like size and dpi
            plt.savefig(
                f"{self.saving_path}\\Plots\\{prefix}_ConfMatrix_{model_name}.png",
                bbox_inches="tight",
            )
        plt.close()

    def plot_training_curves(
        self,
        clf,
        test_label_name,
        eval_score,
        model_name,
        cv_results,
        use_from_cross_val=False,
    ):
        if hasattr(clf, "cv_results_"):
            val_scores = clf.cv_results_[f"mean_test_{eval_score}"]
            train_scores = clf.cv_results_[f"mean_train_{eval_score}"]
        else:
            val_scores = cv_results[f"test_{eval_score}"]
            train_scores = cv_results[f"train_{eval_score}"]

        plt.plot(val_scores, label=test_label_name)
        plt.plot(train_scores, label="train")
        plt.title(f"Validation Curve for {model_name} using {eval_score}.")
        plt.legend(loc="best")
        # plt.show()
        with plt.rc_context():  # Use this to set figure params like size and dpi
            plt.savefig(
                f"{self.saving_path}\\Plots\\TrainingCurve_{model_name}.png",
                bbox_inches="tight",
            )
        plt.close()

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
        selected_features = (
            selector_fitted.get_feature_names_out(data.columns)
            if selector_type == "SequentialFeatureSelector"
            else selector_fitted.get_feature_names_out(X_train.columns)
        )

        return selector_fitted, selected_features

    def feature_engineering(self): ...

    def plot_CV_results(
        self, eval_score, training_plotting_metrics, saving_path, prefix
    ):

        import pandas as pd

        # Extracting data from the dictionary
        names = list(training_plotting_metrics.keys())
        # Get the training/validation results from training_plotting_metrics
        results = [
            result[f"test_{eval_score}"]
            for result in training_plotting_metrics.values()
        ]
        std = [result["std"] for result in training_plotting_metrics.values()]

        # Create a DataFrame
        df_new = pd.DataFrame(
            list(zip(names, results, std)),
            columns=["Model", f"test_{eval_score}", "Std"],
        )

        # Sort the DataFrame
        df_sorted = df_new.sort_values(f"test_{eval_score}")
        df_sorted.index = df_sorted.Model
        df_sorted = df_sorted.round(3)

        df_sorted.rename(
            columns={f"test_{eval_score}": f"training_{eval_score}"}, inplace=True
        )
        # Plotting
        plt.figure(figsize=(15, 10))
        ax = df_sorted.plot(
            kind="barh",
            x="Model",
            y=f"training_{eval_score}",
            xerr="Std",
            facecolor="#AA0000",
            figsize=(15, 10),
            fontsize=12,
            capsize=5,
        )

        gap = 0.015  # Space between the text and the end of the bar
        # You have to call ax.text() for each bar
        # They are already sorted and you need the index of the bar
        for i, (v, s) in enumerate(
            zip(df_sorted[f"training_{eval_score}"], df_sorted.Std)
        ):
            ax.text(
                v + s + gap, i, f"{v} ± {s}", color="blue"
            )  # Place the text at x=v+gap and y= idx

        ax.spines["bottom"].set_color("#CCCCCC")
        ax.set_xlabel(f"Evaluation score used: {eval_score}", fontsize=12)
        ax.set_ylabel("Model", fontsize=12)
        plt.title(f"Model Comparison for Classification in {prefix}")

        # plt.show()
        with plt.rc_context():  # Use this to set figure params like size and dpi
            plt.savefig(
                f"{saving_path}\\Plots\\RankedModelsByMetric.png", bbox_inches="tight"
            )
        # plt.close()

    def plot_clfs(
        self,
        eval_score,
        training_plotting_metrics,
        test_training_plotting_metrics,
        y_test,
        saving_path,
        dummy_score,
    ):

        fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(15, 6))

        # bar chart of accuracy scores
        labels = list(training_plotting_metrics.keys())
        inds = range(1, len(labels) + 1)

        # Cross validation scores
        scores_all = [
            dictionary_scores[f"test_{eval_score}"]
            for acc, dictionary_scores in training_plotting_metrics.items()
        ]

        # Test (unseen) scores
        scores_predictive = [
            dictionary_scores[f"test_{eval_score}"]
            for acc, dictionary_scores in test_training_plotting_metrics.items()
        ]

        ax1.bar(
            inds,
            scores_all,
            color=sns.color_palette(color)[5],
            alpha=0.3,
            hatch="x",
            edgecolor="none",
            label="CrossValidation Set",
        )
        ax1.bar(
            inds,
            scores_predictive,
            color=sns.color_palette(color)[0],
            label="Testing set",
        )
        ax1.set_ylim(0.4, 1)
        ax1.set_ylabel(f"{eval_score} score")
        ax1.axhline(dummy_score, color="black", linestyle="--")
        ax1.set_title(f"{eval_score} scores for basic models", fontsize=17)
        ax1.set_xticks(range(1, len(labels) + 1))
        ax1.set_xticklabels(labels, size=12, rotation=40, ha="right")
        ax1.legend()

        preds_all = [
            dictionary_scores["preds"]
            for acc, dictionary_scores in test_training_plotting_metrics.items()
        ]
        for label, pred in zip(labels, preds_all):
            fpr, tpr, threshold = roc_curve(y_test, pred)
            roc_auc = auc(fpr, tpr)
            ax2.plot(fpr, tpr, label=label + " (area = %0.2f)" % roc_auc, linewidth=2)
        ax2.plot([0, 1], [0, 1], "k--", linewidth=2)
        ax2.set_xlim([-0.05, 1.0])
        ax2.set_ylim([-0.05, 1.05])
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")
        ax2.legend(loc="lower right", prop={"size": 12})
        ax2.set_title("Roc curve for for basic models", fontsize=17)

        # plt.show()
        with plt.rc_context():  # Use this to set figure params like size and dpi
            plt.savefig(f"{saving_path}\\Plots\\AUC.png", bbox_inches="tight")
        # plt.close()

    def shap_analysis(self, model, X_test, X_test_original, y_test):
        import shap

        shap.initjs()

        # Check the type of the model and return the appropriate explainer
        X_test_original_cp = X_test_original.copy()

        for col_type, preprocessor_dict in self.encoder_dict.items():
            if col_type == "categorical_encoder":
                for col, encoder in preprocessor_dict.items():
                    X_test_original_cp = encoder.inverse_transform(X_test_original_cp)
            if col_type == "numerical_encoder":
                for col, encoder in preprocessor_dict.items():
                    X_test_original_cp = encoder.inverse_transform(X_test_original_cp)
        if isinstance(
            model, (LogisticRegression, Perceptron, SGDClassifier, RidgeClassifierCV)
        ):
            explainer = shap.LinearExplainer(model, X_test_original_cp)
        elif isinstance(
            model,
            (
                RandomForestClassifier,
                DecisionTreeClassifier,
                HistGradientBoostingClassifier,
                AdaBoostClassifier,
                XGBClassifier,
                CatBoostClassifier,
                LGBMClassifier,
            ),
        ):
            explainer = shap.TreeExplainer(model)
        elif isinstance(
            model, (KNeighborsClassifier, SVC, ComplementNB, MLPClassifier)
        ):
            explainer = shap.KernelExplainer(model.predict_proba, X_test_original_cp)
        else:
            raise ValueError(f"Unsupported model: {type(model)}")

        explainer_shap = explainer(X_test_original_cp)
        shap_values = explainer.shap_values(X_test_original_cp)
        shap.plots.bar(explainer_shap)
        with plt.rc_context():  # Use this to set figure params like size and dpi
            plt.savefig(
                f"{self.saving_path}\\Plots\\SHAP_Bar_Plot.png", bbox_inches="tight"
            )
        plt.close()

        shap.summary_plot(explainer_shap, X_test_original_cp)
        with plt.rc_context():  # Use this to set figure params like size and dpi
            plt.savefig(
                f"{self.saving_path}\\Plots\\SHAP_Summary_Plot.png", bbox_inches="tight"
            )
            plt.close()

        # Create a SHAP decision plot for the first instance
        shap.decision_plot(
            explainer.expected_value, shap_values[0, :], X_test_original_cp.iloc[0, :]
        )
        with plt.rc_context():
            # Save the plots to the reports folder
            plt.savefig(
                f"{self.saving_path}\\Plots\\SHAP_Decision_Plot.png",
                bbox_inches="tight",
            )
            plt.close()

        # add a force plot
        shap.force_plot(
            explainer.expected_value,
            shap_values[0, :],
            X_test_original_cp.iloc[0, :],
            matplotlib=True,
        )
        # save the force plot with rc
        with plt.rc_context():
            plt.savefig(
                f"{self.saving_path}\\Plots\\SHAP_Force_Plot.png",
                bbox_inches="tight",
            )
            plt.close()

        return shap_values, explainer
