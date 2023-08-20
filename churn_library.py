"""
churn_library.py
This module contains functions used for the churn project.

Author:  Gianluca Manca <gianluca.manca89@gmail.com>
Date:    2023-08-20

Functions:
        import_data: import csv file as pandas dataframe
        add_churn_column: add churn column to dataframe based on Attrition_Flag
        perform_eda: perform exploratory data analysis
        encoder_helper: helper function to turn each categorical column into a
                        new column with propotion of churn for each category
        perform_feature_engineering: perform feature engineering
        train_models: train and store models
        test_model: test and store model results: images + scores
        feature_importance_plot: creates and stores the feature importances in
                        pth
        main: runs main script

Global Variables:
        None
"""


# import libraries
import os
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    classification_report,
    roc_curve,
    auc,
    RocCurveDisplay,
)

import constants as const

# set environment variable to avoid QT related error
os.environ["QT_QPA_PLATFORM"] = "offscreen"


def import_data(pth: str) -> pd.DataFrame or None:
    """
    returns dataframe for the csv found at pth

    Parameters
    ----------
    pth : str
        a path to csv file

    Returns
    -------
    pd.DataFrame or None
        pandas dataframe if valid path else None
    """
    try:
        df = pd.read_csv(pth, index_col=0)
        print("File imported successfully")
        return df
    except FileNotFoundError:
        print("File not found, please check the path and try again")
        return None


def add_churn_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    add churn column to dataframe based on Attrition_Flag column

    Parameters
    ----------
    df : pd.DataFrame
        dataframe to add churn column to

    Returns
    -------
    pd.DataFrame
        dataframe with churn column added
    """
    try:
        df["Churn"] = df["Attrition_Flag"].apply(
            lambda val: 0 if val == "Existing Customer" else 1
        )
        print("Churn column added successfully")
        return df
    except KeyError:
        print("Dataframe does not have Attrition_Flag column")
        return df


def perform_eda(df: pd.DataFrame) -> None:
    """
    perform exploratory data analysis on df and save figures to images folder

    Parameters
    ----------
    df : pd.DataFrame
        dataframe to perform eda on

    Returns
    -------
    None
    """
    pth = "./images/eda"
    # check if image directory already exists
    if os.path.isdir(pth):
        print(f"{pth} directory already exists")
    else:
        # create directory
        os.makedirs(pth)
        print(f"{pth} directory created")

    # create plots
    # plot distribution of individual variables
    for col in df.columns:
        _, ax = plt.subplots()
        ax = sns.histplot(df[col])
        ax.figure.tight_layout()
        plt.savefig(f"{pth}/{col}_dist.png")
        plt.close()

    # plot distribution of individual variables and add a smooth curve obtained
    # using a kernel density estimate
    for col in df.columns:
        _, ax = plt.subplots()
        ax = sns.histplot(df[col], kde=True)
        ax.figure.tight_layout()
        plt.savefig(f"{pth}/{col}_dist_kde.png")
        plt.close()

    # plot correlation heatmap
    # ensure that only quantitative variables are passed to the heatmap
    try:
        _, ax = plt.subplots()
        ax = sns.heatmap(
            df[const.QUANT_COLUMNS].corr(), annot=False, cmap="coolwarm"
        )
        ax.figure.tight_layout()
        plt.savefig(f"{pth}/corr_heatmap.png")
        plt.close()
    except ValueError:
        print(
            "Correlation heatmap could not be generated due to incorrect column types"
        )
    except KeyError:
        print(
            "Correlation heatmap could not be generated due to incorrect column names"
        )

    # plot count of target variable
    try:
        _, ax = plt.subplots()
        ax = sns.countplot(x="Churn", data=df)
        ax.figure.tight_layout()
        plt.savefig(f"{pth}/target_count.png")
        plt.close()
    except ValueError:
        print("Target variable churn could not be found")
    except KeyError:
        print("Target variable churn could not be found")

    # plot distribution of target variable
    try:
        _, ax = plt.subplots()
        ax = sns.histplot(df["Churn"])
        ax.figure.tight_layout()
        plt.savefig(f"{pth}/target_dist.png")
        plt.close()
    except ValueError:
        print("Target variable churn could not be found")
    except KeyError:
        print("Target variable churn could not be found")

    # plot count of target variable grouped by categorical variables
    try:
        for col in df.columns:
            if df[col].dtype == "object":
                _, ax = plt.subplots()
                ax = sns.countplot(x=col, hue="Churn", data=df)
                ax.figure.tight_layout()
                plt.savefig(f"{pth}/{col}_count.png")
                plt.close()
    except ValueError:
        print("Target variable churn could not be found")
    except KeyError:
        print("Target variable churn could not be found")

    # plot distribution of target variable grouped by categorical variables
    try:
        for col in df.columns:
            if df[col].dtype == "object":
                _, ax = plt.subplots()
                ax = sns.histplot(x=col, hue="Churn", data=df)
                ax.figure.tight_layout()
                plt.savefig(f"{pth}/{col}_dist.png")
                plt.close()
    except ValueError:
        print("Target variable churn could not be found")
    except KeyError:
        print("Target variable churn could not be found")

    return None


def encoder_helper(df: pd.DataFrame, response: str = "Churn") -> pd.DataFrame:
    """
    helper function to turn each categorical column into a new column with
    propotion of churn for each category

    Parameters
    ----------
    df : pandas dataframe
    response : str, optional
        string of response name, by default "Churn"

    Returns
    -------
    pd.DataFrame
        dataframe with new columns for each categorical column
    """
    # copy the dataframe
    df = df.copy()
    # iterate over each categorical column
    for col in const.CAT_COLUMNS:
        try:
            # check if the column is in the dataframe
            if col not in df.columns:
                print(f"Column {col} not found")
                continue
            # create a new column name for each category
            new_col = col + "_" + response
            # create a new column with the proportion of churn for each
            # category
            col_groups = df[[col, response]].groupby(col).mean()[response]
            df[new_col] = [col_groups.loc[val] for val in df[col]]
        except KeyError:
            print("Response column not found")
    # drop the original categorical columns
    df.drop(const.CAT_COLUMNS, axis=1, inplace=True)
    # drop the response column
    try:
        df.drop(response, axis=1, inplace=True)
    except KeyError:
        print("Response column not found")
    return df


def perform_feature_engineering(
    df: pd.DataFrame, response: str = "Churn"
) -> list or None:
    """
    perform feature engineering on df and return X_train, X_test, y_train, y_test

    Parameters
    ----------
    df : pandas dataframe
    response : str, optional
        string of response name, by default "Churn"

    Returns
    -------
    list or None
        X_train: X training data
        X_test: X testing data
        y_train: y training data
        y_test: y testing data
    """
    # turn each categorical column into a new column with propotion of churn for
    # each category; only keep the columns we want to use for modeling
    try:
        X = encoder_helper(df, response)[const.KEEP_COLS]
    except KeyError:
        print(
            "The columns you are trying to keep do not exist in the dataframe"
        )
        return None
    # get the target variable
    try:
        y = df[response]
    except KeyError:
        print("The target variable does not exist in the dataframe")
        return None
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=const.TEST_SIZE, random_state=const.RAND_STATE
    )
    return [X_train, X_test, y_train, y_test]


def train_models(
    X_train: pd.DataFrame or np.array, y_train: pd.Series or np.array
) -> list:
    """
    train and store models.

    Parameters
    ----------
    X_train : array-like
        X training data
    y_train : array-like
        y training data

    Returns
    -------
    list
        list of models
    """
    # grid search
    rfc = RandomForestClassifier(random_state=const.RAND_STATE)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(
        solver=const.LR_SOLVER, max_iter=const.LR_MAX_ITER
    )

    # fit models
    cv_rfc = GridSearchCV(
        estimator=rfc, param_grid=const.GRID_PARAM, cv=const.GRID_CV
    )
    cv_rfc.fit(X_train, y_train)
    lrc.fit(X_train, y_train)

    # save models (best estimator for random forest)
    pth = "./models"
    # check if models directory already exists
    if os.path.isdir(pth):
        print(f"{pth} directory already exists")
    else:
        # create directory
        os.makedirs(pth)
        print(f"{pth} directory created")
    joblib.dump(cv_rfc.best_estimator_, f"{pth}/rfc_model.pkl")
    joblib.dump(lrc, f"{pth}/lrc_model.pkl")

    # return models
    return [cv_rfc.best_estimator_, lrc]


def test_model(
    X_train: pd.DataFrame or np.array,
    X_test: pd.DataFrame or np.array,
    y_train: pd.Series or np.array,
    y_test: pd.Series or np.array,
    model: BaseEstimator,
) -> None:
    """
    test and store model results: images + scores

    Parameters
    ----------
    X_train : array-like
        X training data
    X_test : array-like
        X testing data
    y_train : array-like
        y training data
    y_test : array-like
        y testing data
    model : BaseEstimator
        model object

    Returns
    -------
    None
    """
    # predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    pth = "./images/results"
    # check if image directory already exists
    if os.path.isdir(pth):
        print(f"{pth} directory already exists")
    else:
        # create directory
        os.makedirs(pth)
        print(f"{pth} directory created")

    # roc curve image
    fpr, tpr, _ = roc_curve(y_test, y_pred_test)
    roc_auc = auc(fpr, tpr)
    display = RocCurveDisplay(
        fpr=fpr,
        tpr=tpr,
        roc_auc=roc_auc,
        estimator_name=model.__class__.__name__,
    )
    display.plot()
    plt.savefig(f"{pth}/{model.__class__.__name__}_roc_curve.png")
    plt.close()

    # classification report image
    plt.rc("figure", figsize=(8, 5))
    plt.text(
        0.01,
        1.25,
        str(f"{model.__class__.__name__} Train"),
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.text(
        0.01,
        0.05,
        str(classification_report(y_test, y_pred_test)),
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.text(
        0.01,
        0.6,
        str(f"{model.__class__.__name__} Test"),
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.text(
        0.01,
        0.7,
        str(classification_report(y_train, y_pred_train)),
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.axis("off")
    plt.savefig(f"{pth}/{model.__class__.__name__}_classification_report.png")
    plt.close()

    return None


def feature_importance_plot(
    model: BaseEstimator, X_data: pd.DataFrame or np.array
) -> None:
    """
    creates and stores the feature importances in pth

    Parameters
    ----------
    model : BaseEstimator
        classifier model
    X_data : array-like
        input data

    Returns
    -------
    None
    """
    pth = "./images/results"
    # check if image directory already exists
    if os.path.isdir(pth):
        print(f"{pth} directory already exists")
    else:
        # create directory
        os.makedirs(pth)
        print(f"{pth} directory created")

    # calculate feature importance using shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_data)
    shap.summary_plot(shap_values, X_data, plot_type="bar")
    plt.savefig(f"{pth}/{model.__class__.__name__}_feature_importance.png")
    plt.close()

    return None


if __name__ == "__main__":
    # import the dataframe
    df = import_data(r"./data/bank_data.csv")
    print(df.head())
    # add churn column to the dataframe
    df = add_churn_column(df)
    print(df.head())
    # print the shape of the dataframe
    print(f"The shape of the dataframe is: {df.shape}")
    # print sum of null values over all columns in dataframe
    print(
        f"The sum of null values in the dataframe is: {df.isnull().sum().sum()}"
    )
    # print the descriptive statistics of the dataframe
    print("The descriptive statistics of the dataframe is:")
    print(df.describe())
    # perform the exploratory data analysis
    # and save the figures to the images folder
    perform_eda(df)
    # perform feature engineering
    X_train, X_test, y_train, y_test = perform_feature_engineering(df, "Churn")
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    # train models
    rfc, lrc = train_models(X_train, y_train)
    # load models
    rfc = joblib.load("./models/rfc_model.pkl")
    lrc = joblib.load("./models/lrc_model.pkl")
    # test models both with X_train and X_test
    test_model(X_train, X_test, y_train, y_test, rfc)
    test_model(X_train, X_test, y_train, y_test, lrc)
    # get feature importance plots
    feature_importance_plot(rfc, X_test)
