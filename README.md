# EzML: Customizable Machine Learning Pipeline for Tabular Data Classification models

Pipeline for ML on tabular data for classification purposes (regression soon to be added).

# How it works

An orchestrator present in the 'main.py' reads the 'config.json' file which is a configuration file that contains all the information necessary to preprocess, perform Cross-Validation steps, feature selection, as well as models to do training and optimization parameters while saving all the steps in a 'loguru.log' file. These scripts and JSON file can be found on the 'src' folder.

The scripts on the 'src' folder are:

- 'main.py': The main file that runs the orchestrator. This orchestrator in turn calls the remaining steps of the pipeline according to the settings on the 'config.json' configuration file.

- 'preprocessing.py': This is the first script that is run by the orchestrator in order to preprocess the data (scaling, solving missing values, categorical encoding, etc). This script contains the 'PreProcessor' class with the methods necessary to do the preprocessing steps.

- 'train_model.py': A python script containing the 'ModelTraining' class that reads the 'config.json' configuration file, trains the models, optimizes, saves the best model and other plots for model evaluation as well as an excel file with the metrics employed in the cross validation set.

- 'utils.py': A utilities script that contains helper functions independent of the main classes from 'preprocessing.py' and 'train_model.py'.

# Things to consider on the 'config.json' configuration file:

The setting 'path_backbone' is the path for the repository. If you clone the repo to your local system, make sure to paste the path to the local repo clone in this setting. The 'data' setting is where the data is located inside the 'path_backbone' directory. This can be kept as is as long as the structure of the repository isn't changed (particularly naming on the 'data' folder).

# Next features:

New features will be included, such as:

- Configuration for regression problems;
- Settings for both binary, multi-class and multi-label classification problems;
- Improved logging and error catching;
- Better 'readme' file containing detailed information on how to operate the 'config.json' file.
