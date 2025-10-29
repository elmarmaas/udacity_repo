# library doc string


# import libraries
'''The shap library (SHapley Additive exPlanations) is used to explain 
the output of machine learning models. It provides tools to interpret 
model predictions by calculating feature importance values based on 
Shapley values from cooperative game theory. 
This helps you understand how each feature contributes to a 
specific prediction or to the overall model '''
import shap

'''The joblib library is used for efficiently serializing (saving) and 
deserializing (loading) Python objects, especially large data like 
machine learning models or NumPy arrays. It is commonly used to save trained 
models to disk and load them later for predictions or further analysis.'''
import joblib

'''The logging library is in this context used for reporting eda and 
other data during the processing.
It helps developers monitor and debug their code by
recording messages about the program's execution, errors, and other
important information.'''
import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

#from sklearn.metrics import plot_roc_curve, classification_report


import os
os.environ['QT_QPA_PLATFORM']='offscreen'

eda_logger = logging.getLogger('churn.eda')
eda_handler = logging.FileHandler('./logs/churn_eda.log', 
                                  mode='w') #overwrite log file each run
eda_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
eda_handler.setFormatter(eda_formatter)
eda_logger.addHandler(eda_handler)
eda_logger.setLevel(logging.INFO)   


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    try:
        df = pd.read_csv(pth,
                         encoding='utf-8')
        return df
    except FileNotFoundError as err:
        print("File not found", err)
        raise err

def perform_eda(df):
    '''
    perform eda (exploratory data analysis on df and save figures to 
    images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    eda_logger.info("Data preview:\n%s", df.head())
    eda_logger.info("Data dimension: %s",df.shape)
    eda_logger.info("Missing values per column:\n%s", df.isnull().sum())
    eda_logger.info("Data description:\n%s", df.describe())
    # check whether Attrition Flag has two unique values
    eda_logger.info("Unique values in 'Attrition_Flag': %s", 
                    df['Attrition_Flag'].unique())
    if df['Attrition_Flag'].nunique() == 2:
        eda_logger.info("'Attrition_Flag' has two unique values as expected.")
    else:  
        eda_logger.warning("'Attrition_Flag' does not have two unique values."\
        " Check data integrity.")
    # change text in data field to numerical value for churn in order to 
    # prepare data for supervised learning
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" 
        else 1)

    # Create and save the histogram
    plt.figure(figsize=(20,10)) 
    df['Churn'].hist()
    plt.title('Churn Distribution')
    plt.xlabel('Churn (0=Existing, 1=Churned)')
    plt.ylabel('Rate')
    plt.savefig('./images/churn_histogram.png', bbox_inches='tight', dpi=300)
    plt.close()  # Close the figure to free memory
    


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the 
    notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could 
                      be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    pass


def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that 
              could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and s
    tores report as image in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    pass


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    pass

def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    pass

if __name__ == "__main__":
    # call functions according to sequencediagram.jpg
    df = import_data("./data/bank_data.csv")
    perform_eda(df)
    
