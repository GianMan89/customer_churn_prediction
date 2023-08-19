# library doc string
"""
churn_library.py
This module contains functions used for the churn project.

Author:  Gianluca Manca

Functions:
        import_data: import csv file as pandas dataframe
        perform_eda: perform exploratory data analysis
        encoder_helper: helper function to turn each categorical column into a 
                        new column with propotion of churn for each category
        perform_feature_engineering: perform feature engineering
        classification_report_image: produces classification report for 
                        training and testing results and stores report as image
        feature_importance_plot: creates and stores the feature importances in 
                        pth
        train_models: train, store model results: images + scores, and store 
                        models
        main: runs main script

Global Variables:
        None
"""


# import libraries
import os
import pandas as pd

# set environment variable to avoid QT related error
os.environ["QT_QPA_PLATFORM"] = "offscreen"


def import_data(pth: str) -> pd.DataFrame or None:
    """
    returns dataframe for the csv found at pth

    input:
            pth (str): a path to csv file

    output:
            df: pandas dataframe
            None if invalid path
    """
    # check if path is valid
    # if not valid, return None
    # if valid, return dataframe
    try:
        df = pd.read_csv(pth)
        return df
    except FileNotFoundError:
        print("File not found, please check the path and try again")
        return None


def perform_eda(df: pd.DataFrame) -> None:
    """
    perform exploratory data analysis on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    """
    pass


def encoder_helper(df, category_lst, response):
    """
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    """
    pass


def perform_feature_engineering(df, response):
    """
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    """


def classification_report_image(
    y_train,
    y_test,
    y_train_preds_lr,
    y_train_preds_rf,
    y_test_preds_lr,
    y_test_preds_rf,
):
    """
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    """
    pass


def feature_importance_plot(model, X_data, output_pth):
    """
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    """
    pass


def train_models(X_train, X_test, y_train, y_test):
    """
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    """
    pass


if __name__ == "__main__":
    pass
