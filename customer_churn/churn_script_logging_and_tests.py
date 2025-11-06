'''
Module churn_script_logging_and_tests.py
Pytest script for all functions in churn_library.py

Authors: Udacity and Elmar Maas
Date of last update: November 06, 2025
'''

import os
import sys
import logging
import pytest
from pathlib import Path
from collections import namedtuple

# Import machine learning libraries needed for tests
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report

# import churn_library_solution as cls
import churn_library as cls

# Define the namedtuple (same as in churn_library.py)
ClassifierData = namedtuple('ClassifierData',
                            ['y_train',
                             'y_test',
                             'y_train_preds_lr',
                             'y_train_preds_rf',
                             'y_test_preds_lr',
                             'y_test_preds_rf'])


# Enhanced logging configuration for tests
@pytest.fixture(autouse=True)
def setup_test_logging():
    """Setup logging for test execution"""
    # Create logs directory if it doesn't exist
    Path('./logs').mkdir(exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('./logs/churn_library_test.log', mode='w'),
            logging.StreamHandler(sys.stdout)  # Also log to console
        ],
        force=True  # force reconfiguration of existing loggers
    )
    
    # Ensure churn library loggers also output to console
    churn_eda_logger = logging.getLogger('churn.eda')
    churn_lib_logger = logging.getLogger('churn.library')
    
    # Set up test logger
    test_logger = logging.getLogger('pytest_tests')
    test_logger.setLevel(logging.DEBUG)
    
    # Add console handlers if they don't exist
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    if not any(isinstance(h, logging.StreamHandler) for h in churn_eda_logger.handlers):
        churn_eda_logger.addHandler(console_handler)
    
    if not any(isinstance(h, logging.StreamHandler) for h in churn_lib_logger.handlers):
        churn_lib_logger.addHandler(console_handler)

    yield test_logger
        
    # Cleanup: remove console handlers after test
    churn_eda_logger.removeHandler(console_handler)
    churn_lib_logger.removeHandler(console_handler)
    
    # Cleanup
    logging.shutdown()


@pytest.fixture
def clean_images_dir():
    """Fixture to clean up test images after test"""
    
    # Setup: ensure directories exist
    os.makedirs('./images', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)
    
    yield  # Run the test
    # Cleanup after test
    image_files = [
        './images/eda/churn_distribution.png',
        './images/eda/customer_age_distribution.png',
        './images/eda/customer_marital_status_distribution.png',
        './images/eda/total_transition_distribution.png',
        './images/eda/feature_correlation_heatmap.png'
        './images/results/random_forest_classifier_report.png'
        './images/results/logistic_regression_classifier_report.png'
        './images/results/feature_importance_plot.png'
    ]
    for file_path in image_files:
        if os.path.exists(file_path):
            os.remove(file_path)

@pytest.fixture
def sample_dataframe():
    """Fixture to provide sample dataframe for testing"""
    return cls.import_data("./data/bank_data.csv")


def test_import():
    '''
    test data import - test that data is loaded correctly
        and we get the expected number of rows and columns
    '''
    test_logger = logging.getLogger('pytest_tests')
    
    try:
        df = cls.import_data("./data/bank_data.csv")
        test_logger.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        test_logger.error("Testing import_data: The file wasn't found")
        raise err
    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
        test_logger.info("Testing import_data - shape test: SUCCESS")
    except AssertionError as err:
        test_logger.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err

    try:
        test_logger.info(
            "Testing import_data with nonexistent file... \
                - the following error is expected")
        # This should raise FileNotFoundError
        df = cls.import_data("nonexistent_file.csv")
        test_logger.error(
            "Testing import_data: FAILED - No exception for nonexistent file")
    except FileNotFoundError as err:
        test_logger.info("Testing import_data: SUCCESS - caught FileNotFoundError")


def test_perform_eda(sample_dataframe, clean_images_dir):
    '''
    test perform eda function with cleanup
    '''
    test_logger = logging.getLogger('pytest_tests')
    # Perform EDA
    cls.perform_eda(sample_dataframe)

    # Check that EDA images were created
    expected_files = [
        './images/eda/churn_distribution.png',
        './images/eda/customer_age_distribution.png',
        './images/eda/customer_marital_status_distribution.png',
        './images/eda/total_transition_distribution.png',
        './images/eda/feature_correlation_heatmap.png'
    ]

    for file_path in expected_files:
        assert os.path.exists(file_path), f"File {file_path} was not created"
        assert os.path.getsize(file_path) > 0, f"File {file_path} is empty"

    test_logger.info("Testing perform_eda: SUCCESS")

def test_encoder_helper(sample_dataframe):
    '''
    test encoder helper
    '''
    test_logger = logging.getLogger('pytest_tests')
    test_logger.debug("Testing encoder_helper: START")
    category_lst = ['Gender', 'Education_Level', 'Marital_Status', 
                    'Income_Category', 'Card_Category']
    
    # Test encoder helper
    # we need the Churn columns to be added - thus we have to run EDA
    cls.perform_eda(sample_dataframe)
    df_encoded = cls.encoder_helper(sample_dataframe, category_lst, 'Churn')
    test_logger.debug("df_encoded head: %s", df_encoded.head())
    # Check that new columns were created
    for category in category_lst:
        encoded_col = f'{category}_Churn'
        assert encoded_col in df_encoded.columns, f"Column {encoded_col} not found"
        assert df_encoded[encoded_col].dtype in ['float64', 'float32'], \
               f"Column {encoded_col} should be numeric"
    
    test_logger.info("Testing encoder_helper: SUCCESS")


def test_perform_feature_engineering(sample_dataframe):
    '''
    test perform_feature_engineering
    '''
    test_logger = logging.getLogger('pytest_tests')
    test_logger.debug("Testing perform_feature_engineering : START")
    # Test feature engineering
    # prepare data first:
    cls.perform_eda(sample_dataframe)
    X_train, X_test, y_train, y_test = cls.perform_feature_engineering(
        sample_dataframe, 'Churn')
    
    # Check shapes
    assert X_train.shape[0] > 0, "X_train is empty"
    assert X_test.shape[0] > 0, "X_test is empty"
    assert len(y_train) > 0, "y_train is empty"
    assert len(y_test) > 0, "y_test is empty"
    
    # Check that train and test sets have same number of features
    assert X_train.shape[1] == X_test.shape[1], "Train and test feature counts differ"
    
    # Check that split ratio is approximately correct (70/30)
    total_samples = len(y_train) + len(y_test)
    test_ratio = len(y_test) / total_samples
    assert 0.25 <= test_ratio <= 0.35, f"Test ratio {test_ratio} not around 0.3"
    
    test_logger.info("Testing perform_feature_engineering: SUCCESS")


def test_feature_importance_plot(sample_dataframe, clean_images_dir):
    """
    test feature importance plotting
    """
    test_logger = logging.getLogger('pytest_tests')
    test_logger.debug("Testing classification_report_image : START")
    
    # prepare input data for classification report
    cls.perform_eda(sample_dataframe)
    x_train, x_test, y_train, y_test = cls.perform_feature_engineering(sample_dataframe, 'Churn')
    
    rfc = RandomForestClassifier(random_state=42)
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

    # data preparation done - now test the function:    
    cls.feature_importance_plot(cv_rfc.best_estimator_,
                                x_test,
                                './images/results/feature_importance_plot.png')
    expected_files = [
            './images/results/feature_importance_plot.png'
    ]    
    
    for file_path in expected_files:
        assert os.path.exists(file_path), f"File {file_path} was not created"
        assert os.path.getsize(file_path) > 0, f"File {file_path} is empty"

    test_logger.info("Testing feature_importance_plot: SUCCESS")
    
    

def test_classification_report_image(sample_dataframe, clean_images_dir):
    """
    test classification report image
    """
    test_logger = logging.getLogger('pytest_tests')
    test_logger.debug("Testing classification_report_image : START")
    
    # prepare input data for classification report
    cls.perform_eda(sample_dataframe)
    x_train, x_test, y_train, y_test = cls.perform_feature_engineering(sample_dataframe, 'Churn')
    
    rfc = RandomForestClassifier(random_state=42)
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
    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    classifier_data = ClassifierData(y_train=y_train,
                                     y_test=y_test,
                                     y_train_preds_lr=y_train_preds_lr,
                                     y_train_preds_rf=y_train_preds_rf,
                                     y_test_preds_lr=y_test_preds_lr,
                                     y_test_preds_rf=y_test_preds_rf
                                     )
    # now we can test the actual function
    cls.classification_report_image(classifier_data)
    expected_files = [
        './images/results/random_forest_classifier_report.png',
        './images/results/logistic_regression_classifier_report.png'
    ]

    for file_path in expected_files:
        assert os.path.exists(file_path), f"File {file_path} was not created"
        assert os.path.getsize(file_path) > 0, f"File {file_path} is empty"

    test_logger.info("Testing classification_report_image: SUCCESS")


def test_train_models(sample_dataframe):
    '''
    test train_models
    '''
    test_logger = logging.getLogger('pytest_tests')
    test_logger.debug("Testing train_models : START")
    # before we can test the training we have to prepare the data:
    cls.perform_eda(sample_dataframe)
    X_train, X_test, y_train, y_test = cls.perform_feature_engineering(sample_dataframe, 'Churn')
    
    # Train models
    cls.train_models(X_train, X_test, y_train, y_test)
    
    # Check that model files were created
    model_files = [
        './models/rfc_model.pkl',
        './models/logistic_model.pkl'
    ]
    
    for model_file in model_files:
        assert os.path.exists(model_file), f"Model file {model_file} not created"
        assert os.path.getsize(model_file) > 0, f"Model file {model_file} is empty"
    
    # Check that some evaluation images were created
    eval_images = [
        './images/logistic_regression_classifier_report.png',
        './images/random_forest_classifier_report.png'
    ]
    
    for image_file in eval_images:
        if os.path.exists(image_file):  # Some images might be optional
            assert os.path.getsize(image_file) > 0, f"Image file {image_file} is empty"
    
    test_logger.info("Testing train_models: SUCCESS")

# if __name__ == "__main__":
#     test_import(cls.import_data)
#     test_eda(cls.perform_eda, clean_images_dir)
