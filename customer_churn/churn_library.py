# library doc string
'''The churn_library.py is a library of functions to find customers, 
who are likely to churn. It reads customer data from a csv file, 
performs exploratory data analysis (EDA), 
encodes categorical variables, performs feature engineering, trains
machine learning models (logistic regression and random forest), 
evaluates the models, creates explanability plots using SHAP values 
and saves the models to disk.
Processing is completed by providing features importance plots 
and ROC curves, for assisting in estimating future custoemr churn.
Each function is documented with input and output specifications.
'''

# import libraries

import seaborn as sns
import os
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
'''The logging library is in this context used for reporting eda and 
other data during the processing.
It helps developers monitor and debug their code by
recording messages about the program's execution, errors, and other
important information.'''
import logging
'''The joblib library is used for efficiently serializing (saving) and 
deserializing (loading) Python objects, especially large data like 
machine learning models or NumPy arrays. It is commonly used to save trained 
models to disk and load them later for predictions or further analysis.'''
import joblib
'''The shap library (SHapley Additive exPlanations) is used to explain 
the output of machine learning models. It provides tools to interpret 
model predictions by calculating feature importance values based on 
Shapley values from cooperative game theory. 
This helps you understand how each feature contributes to a 
specific prediction or to the overall model '''
import shap
import time

sns.set()

from sklearn.metrics import classification_report

# plot_roc_curve is depricated:
#from sklearn.metrics import plot_roc_curve
# replaced by newer RocCurveDisplay
from sklearn.metrics import RocCurveDisplay

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

eda_logger = logging.getLogger('churn.eda')
eda_handler = logging.FileHandler('./logs/churn_eda.log',
                                  mode='w')  # overwrite log file each run
eda_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
eda_handler.setFormatter(eda_formatter)
eda_logger.addHandler(eda_handler)
eda_logger.setLevel(logging.INFO)

lib_logger = logging.getLogger('churn.library')
lib_handler = logging.FileHandler('./logs/churn_library.log',
                                   mode='w')  # overwrite log file each run
lib_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
lib_handler.setFormatter(lib_formatter)
lib_logger.addHandler(lib_handler)
lib_logger.setLevel(logging.INFO)

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
        eda_logger.error("File not found", err)
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
    eda_logger.info("Data dimension: %s", df.shape)
    eda_logger.info("Missing values per column:\n%s", df.isnull().sum())
    eda_logger.info("Data description:\n%s", df.describe())
    # check whether Attrition Flag has two unique values
    eda_logger.info("Unique values in 'Attrition_Flag': %s",
                    df['Attrition_Flag'].unique())
    if df['Attrition_Flag'].nunique() == 2:
        eda_logger.info("'Attrition_Flag' has two unique values as expected.")
    else:
        eda_logger.warning("'Attrition_Flag' does not have two unique values."
                           " Check data integrity.")
    # change text in data field to numerical value for churn in order to
    # prepare data for supervised learning
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer"
        else 1)

    # Create and save the histogram
    plt.figure(figsize=(20, 10))
    df['Churn'].hist()
    plt.title('Churn Distribution')
    plt.xlabel('Churn (0=Existing, 1=Churned)')
    plt.ylabel('Rate')
    plt.savefig('./images/churn_histogram.png', bbox_inches='tight', dpi=300)
    plt.close()

    plt.figure(figsize=(20, 10))
    df['Customer_Age'].hist()
    plt.title('Customer Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('# of customers')
    plt.savefig('./images/customer_age_histogram.png',
                bbox_inches='tight', dpi=300)
    plt.close()

    plt.figure(figsize=(20, 10))
    marital_counts = df.Marital_Status.value_counts('normalize')
    sns.barplot(x=marital_counts.index,
                y=marital_counts.values,
                hue=marital_counts.index,
                palette='Set1',
                legend=False)  # Disable legend since hue=x makes it redundant
    plt.title('Marital Status')
    plt.xlabel('Marital Status')
    plt.ylabel('Ratio of customers (per status)')
    plt.savefig('./images/customer_marital_status_historgram.png',
                bbox_inches='tight', dpi=300)
    plt.close()

    plt.figure(figsize=(20, 10))
    # distplot is deprecated. Use histplot instead
    # sns.distplot(df['Total_Trans_Ct']);
    # Show distributions of 'Total_Trans_Ct' and add a smooth curve
    # obtained using a kernel density estimate
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    plt.title('Transitions')
    plt.xlabel('Total_Trans_Ct')
    plt.ylabel('Density')
    plt.savefig('./images/density_vs_total_trans_ct_historgram.png',
                bbox_inches='tight', dpi=300)
    plt.close()

    plt.figure(figsize=(20, 10))
    plt.title('Feature Correlation Heatmap')
    # for heatmap only numeric columns can be used:
    numeric_df = df.select_dtypes(include=[np.number])
    # use a sequential color map for the heatmap such that correlation
    # is more intuitively visible
    sns.heatmap(numeric_df.corr(),
                annot=False, cmap='viridis', linewidths=2)
    plt.savefig('./images/feature_correlation_heatmap.png',
                bbox_inches='tight', dpi=300)
    plt.close()


def encoder_helper(df, category_lst, response='Churn'):
    '''
    helper function to turn each categorical column into a new column with
    proportion of churn for each category - associated with cell 15 from the 
    notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that names
            the target category for the mean encoding - default is 'Churn']

    output:
            df_encoded: pandas dataframe with new columns for mean encoded
                        categorical features in extra columns
    '''
    df_encoded = df.copy()
    for category in category_lst:
        cat_name = category + '_' + response
        # only calculate the mean for the passed response column
        # mean() only works for numeric columns which results in errors
        # if we request non numeric response columns
        # this is the easily readable but inefficient way:
        # cat_lst = []
        # cat_groups = df_encoded.groupby(category)[response].mean()
        # for val in df_encoded[category]:
        #     cat_lst.append(cat_groups.loc[val])
        # df_encoded[cat_name] = cat_lst
        # more efficient way:
        df_encoded[cat_name] = df_encoded.groupby(
            category)[response].transform('mean')
    eda_logger.info("Encoded dataframe with focus %s, head:\n%s",
                    response, df_encoded.head())
    return df_encoded


def perform_feature_engineering(df, response='Churn'):
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
    y = df[response]
    X = pd.DataFrame()
    category_lst = ['Gender', 'Education_Level', 'Marital_Status',
                    'Income_Category', 'Card_Category']
    df_encoded = encoder_helper(df, category_lst, 'Churn')
    keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
                 'Total_Relationship_Count', 'Months_Inactive_12_mon',
                 'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
                 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
                 'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
                 'Income_Category_Churn', 'Card_Category_Churn']
    X[keep_cols] = df_encoded[keep_cols]
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.3,
                                                        random_state=42)
    eda_logger.info("Performed feature engineering and split data."
                    " X_train shape: %s, X_test shape: %s",
                    X_train.shape, X_test.shape)
    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and 
    stores report as image in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None (as return parameter)
             But there is a side-effect output: The classification report 
             is saved as image files in images folder
    '''
    # Random Forest Report  
    plt.figure(figsize=(8, 10))  # increased figure size w.r.t. notebook
    plt.text(0.01, 0.9, 'Random Forest Train', 
             {'fontsize': 12}, fontweight='bold')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), 
             {'fontsize': 10}, fontproperties='monospace')
    
    plt.text(0.01, 0.5, 'Random Forest Test', 
             {'fontsize': 12}, fontweight='bold')
    plt.text(0.01, 0.3, str(classification_report(y_test, y_test_preds_rf)), 
             {'fontsize': 10}, fontproperties='monospace')
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)  # Set explicit limits
    plt.axis('off')
    plt.savefig('./images/random_forest_classifier_report.png', 
                bbox_inches='tight', dpi=300)
    plt.close()

    # Logistic Regression Report
    plt.figure(figsize=(8, 10))  # Larger figure size
    plt.text(0.01, 0.9, 'Logistic Regression Train', 
             {'fontsize': 12}, fontweight='bold')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_lr)), 
             {'fontsize': 10}, fontproperties='monospace')
    
    plt.text(0.01, 0.5, 'Logistic Regression Test', 
             {'fontsize': 12}, fontweight='bold')
    plt.text(0.01, 0.3, str(classification_report(y_test, y_test_preds_lr)), 
             {'fontsize': 10}, fontproperties='monospace')
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)  # Set explicit limits
    plt.axis('off')
    plt.savefig('./images/logistic_regression_classifier_report.png', 
                bbox_inches='tight', dpi=300)
    plt.close()


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figures

    output:
             None
    '''
    SHAP_explanation_image_path = output_pth.replace('.png',
                                                  '_shap_summary.png')
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_data)
    shap.summary_plot(shap_values, X_data, plot_type="bar", show=False)
    plt.savefig(SHAP_explanation_image_path, bbox_inches='tight', dpi=300)
    plt.close()

    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20,5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth, bbox_inches='tight', dpi=300)
    plt.close()

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
    starttime = time.time() # for measuring training time
    # grid search
    rfc = RandomForestClassifier(random_state=42)

    # Default solver 'lbfgs' failed to converge thus we use 
    # liblinear solver instead which is good for small (~10k samples) datasets:
    lrc = LogisticRegression(solver='liblinear', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    elapsed = time.time() - starttime
    lib_logger.info("Model training completed in %.4f seconds", elapsed)
    # make predictions
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # scores
    lib_logger.info('random forest results')
    lib_logger.info('test results')
    lib_logger.info(classification_report(y_test, y_test_preds_rf))
    lib_logger.info('train results')
    lib_logger.info(classification_report(y_train, y_train_preds_rf))

    lib_logger.info('logistic regression results')
    lib_logger.info('test results')
    lib_logger.info(classification_report(y_test, y_test_preds_lr))
    lib_logger.info('train results')
    lib_logger.info(classification_report(y_train, y_train_preds_lr))
    
    
    # Replaced deprecated plot_roc_curve with RocCurveDisplay
    # lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    RocCurveDisplay.from_estimator(lrc, X_test, y_test)
    plt.savefig('./images/roc_curve_logistic_regression.png', 
                bbox_inches='tight', dpi=300)
    
    # now plot both curves into the same figure
    plt.figure(figsize=(15, 6))
    ax = plt.gca()  # Get current axes

    # Plot Logistic Regression ROC with blue color
    RocCurveDisplay.from_estimator(lrc, X_test, y_test, ax=ax, alpha=0.8, 
                                color='red', name='Logistic Regression')

    # Plot Random Forest ROC with red color  
    RocCurveDisplay.from_estimator(cv_rfc.best_estimator_, X_test, y_test, 
                                    ax=ax, alpha=0.8,
                                    color='blue', name='Random Forest')

    plt.title('ROC Curves Comparison')
    plt.legend()  # Show legend to distinguish the curves
    plt.grid(True, alpha=0.3)  # Optional: add grid for better readability
    plt.tight_layout()
    plt.savefig('./images/roc_curves_comparison.png', 
                bbox_inches='tight', dpi=300)
    plt.close()

    # save best model with hyperparameter tuning:
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    # no hyperparameter tuning for logistic regression implemented - hence
    # save the model as is:
    joblib.dump(lrc, './models/logistic_model.pkl')
    
    # create images for classification report
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    # trigger feature importance plot
    feature_importance_plot(cv_rfc.best_estimator_,
                            X_test,
                            './images/feature_importance.png')
    
    

if __name__ == "__main__":
    # call functions according to sequencediagram.jpg
    df = import_data("./data/bank_data.csv")
    perform_eda(df)
    X_train, X_test, y_train, y_test = perform_feature_engineering(df, 'Churn')
    train_models(X_train, X_test, y_train, y_test)
