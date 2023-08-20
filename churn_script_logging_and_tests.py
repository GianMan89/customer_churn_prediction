"""
churn_script_logging_and_tests.py
This script contains the tests for the churn_library.py script.

Author:  Gianluca Manca <gianluca.manca89@gmail.com>
Date:    2023-08-20

Functions:
        test_import: test data import
		test_add_churn_column: test add_churn_column
		test_eda: test perform eda function
		test_encoder_helper: test encoder helper
		test_perform_feature_engineering: test perform_feature_engineering
		test_train_models: test train_models
		test_test_model: test test_model
		test_feature_importance_plot: test feature_importance_plot
        main: runs the main script

Global Variables:
        None
"""

import logging
import churn_library as cl

logging.basicConfig(
    filename="./logs/churn_library.log",
    level=logging.INFO,
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
)


def test_import(import_data):
    """
    test data import - this example is completed for you to assist with the
    other test functions
    """
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
        return df
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns"
        )
        raise err


def test_add_churn_column(add_churn_column, df):
    """
    test add_churn_column
    """
    try:
        df = add_churn_column(df)
        logging.info("Testing add_churn_column: SUCCESS")
        return df
    except KeyError as err:
        logging.error(
            "Testing add_churn_column: Dataframe does not have Attrition_Flag column"
        )
        raise err


def test_eda(perform_eda, df):
    """
    test perform eda function
    """
    try:
        perform_eda(df)
        logging.info("Testing perform_eda: SUCCESS")
    except ValueError as err:
        logging.error(
            "Testing perform_eda: Target variable could not be found"
        )
        raise err
    except KeyError as err:
        logging.error(
            "Testing perform_eda: Target variable could not be found"
        )
        raise err


def test_encoder_helper(encoder_helper, df):
    """
    test encoder helper
    """
    try:
        encoder_helper(df)
        logging.info("Testing encoder_helper: SUCCESS")
    except KeyError as err:
        logging.error("Testing encoder_helper: Response column not found")
        raise err


def test_perform_feature_engineering(perform_feature_engineering, df):
    """
    test perform_feature_engineering
    """
    try:
        X_train, X_test, y_train, y_test = perform_feature_engineering(df)
        logging.info("Testing perform_feature_engineering: SUCCESS")
        return [X_train, X_test, y_train, y_test]
    except KeyError as err:
        logging.error("Testing perform_feature_engineering: Columns not found")
        raise err


def test_train_models(train_models, X_train, y_train):
    """
    test train_models
    """
    try:
        rfc, lrc = train_models(X_train, y_train)
        logging.info("Testing train_models: SUCCESS")
        return [rfc, lrc]
    except Exception as err:
        logging.error("Testing train_models: Training unsuccessful")
        raise err


def test_test_model(test_model, X_train, X_test, y_train, y_test, model):
    """
    test test_model
    """
    try:
        test_model(X_train, X_test, y_train, y_test, model)
        logging.info("Testing test_model: SUCCESS")
    except Exception as err:
        logging.error("Testing test_model: Testing unsuccessful")
        raise err


def test_feature_importance_plot(feature_importance_plot, model, X_test):
    """
    test feature_importance_plot
    """
    try:
        feature_importance_plot(model, X_test)
        logging.info("Testing feature_importance_plot: SUCCESS")
    except Exception as err:
        logging.error("Testing feature_importance_plot: Plot unsuccessful")
        raise err


if __name__ == "__main__":
    data = test_import(cl.import_data)
    data = test_add_churn_column(cl.add_churn_column, data)
    test_eda(cl.perform_eda, data)
    X_train, X_test, y_train, y_test = test_perform_feature_engineering(
        cl.perform_feature_engineering, data
    )
    rf_classifier, lr_classifier = test_train_models(
        cl.train_models, X_train, y_train
    )
    test_test_model(
        cl.test_model, X_train, X_test, y_train, y_test, rf_classifier
    )
    test_test_model(
        cl.test_model, X_train, X_test, y_train, y_test, lr_classifier
    )
    test_feature_importance_plot(
        cl.feature_importance_plot, rf_classifier, X_test
    )
