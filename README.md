# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
In this project, we will build a machine learning model to predict customer churn. We will use the **Credit Card Customers** dataset from Kaggle (https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers) to train and test our model. The dataset contains 10,127 customers and 21 features. Each row represents a customer, each column contains customer’s attributes described on the column Metadata. We use the **Attrition_Flag** column to engineer our target variable **Churn**.

We have only **16.07%** of customers who have churned. Thus, it is a challenge to train our model to predict churning customers. 


## Files and data description
Overview of the files and data present in the root directory.
```
├── churn_library.py
├── churn_notebook.ipynb
├── churn_notebook_refactored.ipynb
├── churn_script_logging_and_tests.py
├── constants.py
├── data
│   ├── bank_data.csv
├── .gitignore
├── images
│   ├── eda
│   │   ├── ... set of images for exploratory data analysis
│   ├── results
│   │   ├── ... set of images for model evaluation
├── LICENSE
├── logs
│   ├── churn_library.log
├── models
│   ├── lrc.pkl
│   └── rfc_model.pkl
├── poetry.lock
├── pyproject.toml
└── README.md
```

There are 4 main folders in the repository:

1. `data`: customer churn data saved in `csv` format
2. `images`: stores the results of the exploratory data analysis (subfolder `eda`) and model evaluation (subfolder `results`).
3. `logs`: stores the logs of function test results on the `churn_library.py` file
4. `models`: stores the pickled model objects.

Other important files in this repository include:

1. `churn_library.py`: contains the functions to load the data, perform exploratory data analysis, train and evaluate the models, and save the model objects.
2. `churn_notebook.ipynb`: contains the code to perform exploratory data analysis, train and evaluate the models, and save the model objects.
3. `churn_notebook_refactored.ipynb`: contains the refactored code from `churn_notebook.ipynb` to perform exploratory data analysis, train and evaluate the models, and save the model objects.
4. `churn_script_logging_and_tests.py`: contains the code to perform function tests on the functions in `churn_library.py` and save the logs in `logs/churn_library.log`.
5. `constants.py`: contains the constants used in the `churn_library.py` file.
6. `poetry.lock` and `pyproject.toml`: contains the dependencies used in this project.

## Running Files
1. Clone the repository and navigate to the downloaded folder.
```
git clone
cd customer-churn-prediction
```
2. Create and activate a new environment.
```
python -m venv .venv
```
3. Install required packages from the `poetry.lock` and `pyproject.toml` files.
```
poetry install
```
4. Run the code to train and evaluate a **Linear Regression model** and a **Random Forest Classifier model**.
```
python churn_library.py
```
5. Alternatively, use the Jupyter Notebook `churn_notebook_refactored.ipynb`
```
jupyter notebook churn_notebook_refactored.ipynb
```
6. For testing and logging, run the script `churn_script_logging_and_tests.py`
```
python churn_script_logging_and_tests.py
```



