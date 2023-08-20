"""
constants.py
This file contains the constants used in the project.

Author:  Gianluca Manca <gianluca.manca89@gmail.com>
Date:    2023-08-20

Functions:
        None

Global Variables:
        CAT_COLUMNS: list of categorical columns
        QUANT_COLUMNS: list of quantitative columns
        KEEP_COLS: list of columns to keep
        TEST_SIZE: test size for train test split
        RAND_STATE: random state for train test split
        LR_SOLVER: solver for logistic regression
        LR_MAX_ITER: max iterations for logistic regression
        GRID_CV: number of folds for grid search
        GRID_PARAM: parameters for grid search
"""

CAT_COLUMNS = [
    "Gender",
    "Education_Level",
    "Marital_Status",
    "Income_Category",
    "Card_Category",
]

QUANT_COLUMNS = [
    "Customer_Age",
    "Dependent_count",
    "Months_on_book",
    "Total_Relationship_Count",
    "Months_Inactive_12_mon",
    "Contacts_Count_12_mon",
    "Credit_Limit",
    "Total_Revolving_Bal",
    "Avg_Open_To_Buy",
    "Total_Amt_Chng_Q4_Q1",
    "Total_Trans_Amt",
    "Total_Trans_Ct",
    "Total_Ct_Chng_Q4_Q1",
    "Avg_Utilization_Ratio",
]

KEEP_COLS = [
    "Customer_Age",
    "Dependent_count",
    "Months_on_book",
    "Total_Relationship_Count",
    "Months_Inactive_12_mon",
    "Contacts_Count_12_mon",
    "Credit_Limit",
    "Total_Revolving_Bal",
    "Avg_Open_To_Buy",
    "Total_Amt_Chng_Q4_Q1",
    "Total_Trans_Amt",
    "Total_Trans_Ct",
    "Total_Ct_Chng_Q4_Q1",
    "Avg_Utilization_Ratio",
    "Gender_Churn",
    "Education_Level_Churn",
    "Marital_Status_Churn",
    "Income_Category_Churn",
    "Card_Category_Churn",
]

TEST_SIZE = 0.3
RAND_STATE = 42
LR_SOLVER = "lbfgs"
LR_MAX_ITER = 3000
GRID_CV = 5

GRID_PARAM = {
    "n_estimators": [200, 500],
    "max_features": ["log2", "sqrt"],
    "max_depth": [4, 5, 100],
    "criterion": ["gini", "entropy"],
}
