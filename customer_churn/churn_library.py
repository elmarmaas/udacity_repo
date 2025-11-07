# library doc string
'''
The churn_library.py is a library of functions to find customers,
who are likely to churn. It reads customer data from a csv file,
performs exploratory data analysis (EDA),
encodes categorical variables, performs feature engineering, trains
machine learning models (logistic regression and random forest),
evaluates the models, creates explanability plots using SHAP values
and saves the models to disk.
Processing is completed by providing features importance plots
and ROC curves, for assisting in estimating future customer churn.
Each function is documented with input and output specifications.

Authors: Udacity and Elmar Maas
Date of last update: November 05, 2025
'''

# import libraries
import time
import logging
import os

from collections import namedtuple

# The joblib library is used for efficiently serializing (saving) and
# deserializing (loading) Python objects, especially large data like
# machine learning models or NumPy arrays. It is commonly used to save trained
# models to disk and load them later for predictions or further analysis.
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# The shap library (SHapley Additive exPlanations) is used to explain
# the output of machine learning models. It provides tools to interpret
# model predictions by calculating feature importance values based on
# Shapley values from cooperative game theory.
# This helps you understand how each feature contributes to a
# specific prediction or to the overall model
import shap

# plot_roc_curve is depricated:
# from sklearn.metrics import plot_roc_curve
# replaced by newer RocCurveDisplay:
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# set seaborn theme for plots
sns.set_theme()

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

ClassifierData = namedtuple('ClassifierData',
                            ['y_train',
                             'y_test',
                             'y_train_preds_lr',
                             'y_train_preds_rf',
                             'y_test_preds_lr',
                             'y_test_preds_rf'])

# set up logging
if not os.path.exists('logs'):
    os.makedirs('logs')

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
        df_imported = pd.read_csv(pth,
                                  encoding='utf-8')
        return df_imported
    except FileNotFoundError as err:
        lib_logger.error("File not found: %s", err)
        raise err
    except Exception as err:
        lib_logger.error("Error importing data: %s", err)
        raise err


def perform_eda(eda_df):
    '''
    perform eda (exploratory data analysis) on eda_df and save figures to
    images folder
    input:
            eda_df: pandas dataframe

    output:
            None

    side effect (output):
            eda images are saved to images folder
    '''
    eda_logger.info("Data preview:\n%s", eda_df.head())
    eda_logger.info("Data dimension: %s", eda_df.shape)
    eda_logger.info("Missing values per column:\n%s", eda_df.isnull().sum())
    eda_logger.info("Data description:\n%s", eda_df.describe())
    # check whether Attrition Flag has two unique values
    eda_logger.info("Unique values in 'Attrition_Flag': %s",
                    eda_df['Attrition_Flag'].unique())
    if eda_df['Attrition_Flag'].nunique() == 2:
        eda_logger.info("'Attrition_Flag' has two unique values as expected.")
    else:
        eda_logger.warning("'Attrition_Flag' does not have two unique values."
                           " Check data integrity.")
    # change text in data field to numerical value for churn in order to
    # prepare data for supervised learning
    eda_df['Churn'] = eda_df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer"
        else 1)

    # Create and save the histogram
    plt.figure(figsize=(20, 10))
    eda_df['Churn'].hist()
    plt.title('Churn Distribution')
    plt.xlabel('Churn (0=Existing, 1=Churned)')
    plt.ylabel('Rate')
    plt.savefig('./images/eda/churn_distribution.png',
                bbox_inches='tight', dpi=300)
    plt.close()

    plt.figure(figsize=(20, 10))
    eda_df['Customer_Age'].hist()
    plt.title('Customer Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('# of customers')
    plt.savefig('./images/eda/customer_age_distribution.png',
                bbox_inches='tight', dpi=300)
    plt.close()

    plt.figure(figsize=(20, 10))
    marital_counts = eda_df.Marital_Status.value_counts('normalize')
    sns.barplot(x=marital_counts.index,
                y=marital_counts.values,
                hue=marital_counts.index,
                palette='Set1',
                legend=False)  # Disable legend since hue=x makes it redundant
    plt.title('Marital Status')
    plt.xlabel('Marital Status')
    plt.ylabel('Ratio of customers (per status)')
    plt.savefig('./images/eda/customer_marital_status_distribution.png',
                bbox_inches='tight', dpi=300)
    plt.close()

    plt.figure(figsize=(20, 10))
    # distplot is deprecated. Use histplot instead
    # sns.distplot(eda_df['Total_Trans_Ct']);
    # Show distributions of 'Total_Trans_Ct' and add a smooth curve
    # obtained using a kernel density estimate
    sns.histplot(eda_df['Total_Trans_Ct'], stat='density', kde=True)
    plt.title('Transitions')
    plt.xlabel('Total_Trans_Ct')
    plt.ylabel('Density')
    plt.savefig('./images/eda/total_transition_distribution.png',
                bbox_inches='tight', dpi=300)
    plt.close()

    plt.figure(figsize=(20, 10))
    plt.title('Feature Correlation Heatmap')
    # for heatmap only numeric columns can be used:
    numeric_eda_df = eda_df.select_dtypes(include=[np.number])
    # use a sequential color map for the heatmap such that correlation
    # is more intuitively visible
    sns.heatmap(numeric_eda_df.corr(),
                annot=False, cmap='viridis', linewidths=2)
    plt.savefig('./images/eda/feature_correlation_heatmap.png',
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
    eda_logger.debug("Encoded dataframe with focus %s, head:\n%s",
                     response, df_encoded.head())
    return df_encoded


def perform_feature_engineering(df, response='Churn'):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that
              could be used for naming variables or index y column]

    output:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    y_df = df[response]
    x_df = pd.DataFrame()
    category_lst = ['Gender', 'Education_Level', 'Marital_Status',
                    'Income_Category', 'Card_Category']
    df_encoded = encoder_helper(df, category_lst, 'Churn')
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']
    x_df[keep_cols] = df_encoded[keep_cols]
    x_df_train, x_df_test, y_df_train, y_df_test = train_test_split(
        x_df, y_df, test_size=0.3, random_state=42)
    eda_logger.info("Performed feature engineering and split data."
                    " x_train shape: %s, x_test shape: %s",
                    x_df_train.shape, x_df_test.shape)
    return x_df_train, x_df_test, y_df_train, y_df_test


def classification_report_image(cf_dat: ClassifierData):
    '''
    produces classification report for training and testing results and
    stores report as image in images folder
    input:
            cf_dat: ClassifierData named tuple containing:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None (as return parameter)

    side effect (output):
             The classification report
             is saved as image files in images folder
    '''

    # Random Forest Report
    plt.figure(figsize=(8, 10))  # increased figure size w.r.t. notebook
    plt.text(0.01, 0.9, 'Random Forest Train',
             {'fontsize': 12}, fontweight='bold')
    plt.text(0.01, 0.7, str(classification_report(cf_dat.y_train,
                                                  cf_dat.y_train_preds_rf)),
             {'fontsize': 10}, fontproperties='monospace')

    plt.text(0.01, 0.5, 'Random Forest Test',
             {'fontsize': 12}, fontweight='bold')
    plt.text(0.01, 0.3, str(classification_report(cf_dat.y_test,
                                                  cf_dat.y_test_preds_rf)),
             {'fontsize': 10}, fontproperties='monospace')

    plt.xlim(0, 1)
    plt.ylim(0, 1)  # Set explicit limits
    plt.axis('off')
    plt.savefig('./images/results/random_forest_classifier_report.png',
                bbox_inches='tight', dpi=300)
    plt.close()

    # Logistic Regression Report
    plt.figure(figsize=(8, 10))  # Larger figure size
    plt.text(0.01, 0.9, 'Logistic Regression Train',
             {'fontsize': 12}, fontweight='bold')
    plt.text(
        0.01, 0.7, str(
            classification_report(
                cf_dat.y_train, cf_dat.y_train_preds_lr)), {
            'fontsize': 10}, fontproperties='monospace')

    plt.text(0.01, 0.5, 'Logistic Regression Test',
             {'fontsize': 12}, fontweight='bold')
    plt.text(
        0.01, 0.3, str(
            classification_report(
                cf_dat.y_test, cf_dat.y_test_preds_lr)), {
            'fontsize': 10}, fontproperties='monospace')

    plt.xlim(0, 1)
    plt.ylim(0, 1)  # Set explicit limits
    plt.axis('off')
    plt.savefig('./images/results/logistic_regression_classifier_report.png',
                bbox_inches='tight', dpi=300)
    plt.close()


def feature_importance_plot(model, x_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            x_data : pandas dataframe of X values
            output_pth: path to store the figures

    output:
             None

    side effect (output):
             The feature importance plot is saved to the specified folder
    '''

    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(x_data .shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(x_data .shape[1]), names, rotation=90)
    plt.savefig(output_pth, bbox_inches='tight', dpi=300)
    plt.close()


def shap_explanation_plot(model, x_data, output_pth):
    '''
    creates and stores the SHAP explanation plot in pth
    input:
            model: model object
            x_data : pandas dataframe of X values
            output_pth: path to store the figures

    output:
             None
    side effect (output):
             The shap explanation plot is saved to the specified folder
    '''
    # SHAP explanation plot
    try:
        lib_logger.info("Starting SHAP explanation...")
        lib_logger.info("x_data shape: %s", x_data.shape)
        lib_logger.info("Model type: %s", type(model))
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_data)

        plt.figure(figsize=(20, 20))
        shap.summary_plot(shap_values,
                          x_data,
                          plot_type="bar",
                          show=False)
        plt.savefig(output_pth, bbox_inches='tight', dpi=300)
        plt.close()
    except ValueError as e:
        # Most common: data shape mismatches, invalid parameters
        lib_logger.error("SHAP ValueError (data/parameter issue): %s", e)
    except IndexError as e:
        # Your current issue: array index problems
        lib_logger.error("SHAP IndexError (array access issue): %s", e)
    except AttributeError as e:
        # Missing attributes or methods
        lib_logger.error("SHAP AttributeError (missing attribute): %s", e)
    except TypeError as e:
        # Wrong data types passed
        lib_logger.error("SHAP TypeError (wrong data type): %s", e)
    except MemoryError as e:
        # Large datasets causing memory issues
        lib_logger.error("SHAP MemoryError (insufficient memory): %s", e)
    except Exception as exc:  # pylint: disable=broad-except
        lib_logger.error("Unexpected SHAP error (%s): %s",
                         type(exc).__name__, exc)
        # Log full traceback for debugging
        lib_logger.debug("Full traceback:", exc_info=True)
    # intentionally continue processing even if SHAP plot fails


def roc_curve_report_image(cv_rfc, lrc, x_test, y_test):
    '''
    creates and stores a combined ROC curve plot for
    two models into images folder
    input:
            cv_rfc: RandomForestClassifier model after hyperparameter tuning
            lrc: LogisticRegression model
            x_test: X testing data
            y_test: y testing data

    output:
             None

    side effect (output):
             The roc curve plot is saved to images folder
    '''
    # now plot both curves into the same figure
    plt.figure(figsize=(15, 6))
    ax = plt.gca()  # Get current axes

    # Replaced deprecated plot_roc_curve from notebook with RocCurveDisplay
    # Plot Logistic Regression ROC with blue color
    RocCurveDisplay.from_estimator(lrc, x_test, y_test, ax=ax, alpha=0.8,
                                   color='red', name='Logistic Regression')

    # Plot Random Forest ROC with red color
    RocCurveDisplay.from_estimator(cv_rfc.best_estimator_, x_test, y_test,
                                   ax=ax, alpha=0.8,
                                   color='blue', name='Random Forest')

    plt.title('ROC Curves Comparison')
    plt.legend()  # Show legend to distinguish the curves
    plt.grid(True, alpha=0.3)  # Optional: add grid for better readability
    plt.tight_layout()
    plt.savefig('./images/results/roc_curves_comparison.png',
                bbox_inches='tight', dpi=300)
    plt.close()


def train_models(x_train, x_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None

    side effect (output):
             Several plots are generated in folder images
             and models are saved in folder models
    '''
    starttime = time.time()  # for measuring training time
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
    cv_rfc.fit(x_train, y_train)

    lrc.fit(x_train, y_train)

    elapsed = time.time() - starttime
    lib_logger.info("Model training completed in %.4f seconds", elapsed)

    # make predictions and store in claissifier_data named tuple
    classifier_data = ClassifierData(
        y_train=y_train,
        y_test=y_test,
        y_train_preds_lr=lrc.predict(x_train),
        y_train_preds_rf=cv_rfc.best_estimator_.predict(x_train),
        y_test_preds_lr=lrc.predict(x_test),
        y_test_preds_rf=cv_rfc.best_estimator_.predict(x_test))
    # scores
    lib_logger.info('random forest results')
    lib_logger.info('test results')
    lib_logger.info(
        classification_report(
            y_test,
            classifier_data.y_test_preds_rf))
    lib_logger.info('train results')
    lib_logger.info(
        classification_report(
            y_train,
            classifier_data.y_train_preds_rf))

    lib_logger.info('logistic regression results')
    lib_logger.info('test results')
    # lib_logger.info(classification_report(y_test, y_test_preds_lr))
    lib_logger.info(classification_report(y_test,
                                          classifier_data.y_test_preds_lr))
    lib_logger.info('train results')
    lib_logger.info(classification_report(y_train,
                                          classifier_data.y_train_preds_lr))

    # ROC curves combined plot:
    roc_curve_report_image(cv_rfc, lrc, x_test, y_test)

    try:
        os.makedirs('./models', exist_ok=True)
    except OSError as exc:
        lib_logger.error("Error creating models directory: %s", exc)
        raise exc
    try:
        # save best model with hyperparameter tuning:
        joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
        # no hyperparameter tuning for logistic regression implemented - hence
        # save the model as is:
        joblib.dump(lrc, './models/logistic_model.pkl')
    except OSError as exc:
        lib_logger.error("Error saving model files: %s", exc)
        raise exc

    # create images for classification report
    classification_report_image(classifier_data)

    shap_explanation_plot(cv_rfc.best_estimator_,
                          x_test,
                          './images/results/feature_shap_explanation.png')

    # trigger feature importance plot
    feature_importance_plot(cv_rfc.best_estimator_,
                            x_test,
                            './images/results/feature_importance_plot.png')


def main():
    '''
    Main function that executes the complete churn analysis pipeline.

    This function orchestrates the entire customer churn prediction workflow:
    1. Import data from CSV file
    2. Perform exploratory data analysis (EDA)
    3. Perform feature engineering and data splitting
    4. Train machine learning models (Random Forest and Logistic Regression)
    5. Generate evaluation plots and save trained models

    Input files:
        - ./data/bank_data.csv: Customer data file

    Output files:
        - ./images/: Various analysis and model evaluation plots
        - ./models/: Trained model files (rfc_model.pkl, logistic_model.pkl)
        - ./logs/: Application log files

    Raises:
        FileNotFoundError: If the input data file is not found
        ValueError: If data processing encounters invalid data
        Exception: For any other unexpected errors during processing

    Returns:
        None
    '''
    try:
        # Call functions according to sequence diagram
        bank_df = import_data("./data/bank_data.csv")
        perform_eda(bank_df)
        lib_logger.debug("bank_df after EDA: %s", bank_df.head())
        bank_df_split = perform_feature_engineering(bank_df, 'Churn')
        train_models(bank_df_split[0],
                     bank_df_split[1],
                     bank_df_split[2],
                     bank_df_split[3])
        lib_logger.info("Churn analysis pipeline completed successfully")

    except FileNotFoundError as exc:
        lib_logger.error("Data file not found: %s", exc)
        raise
    except (ValueError, KeyError) as exc:
        lib_logger.error("Data processing error: %s", exc)
        raise
    except Exception as exc:  # pylint: disable=broad-except
        lib_logger.error("Unexpected error in main pipeline: %s", exc)
        raise


if __name__ == "__main__":
    main()
