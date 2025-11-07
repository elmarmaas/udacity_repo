'''
Module churn_script_logging_and_tests.py
Pytest script for all functions in churn_library.py

Usage: For usage of the tests please refer to the README.md file.

Authors: Udacity and Elmar Maas
Date of last update: November 06, 2025
'''
# standard imports
import os
import sys
import logging
import datetime
from pathlib import Path
from collections import namedtuple

# third party imports
import pytest
# Import machine learning libraries needed for tests
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# import churn_library_solution as cls
import churn_library as cls

# Global test configuration - Expected output files
EDA_IMAGE_FILES = [
    './images/eda/churn_distribution.png',
    './images/eda/customer_age_distribution.png',
    './images/eda/customer_marital_status_distribution.png',
    './images/eda/total_transition_distribution.png',
    './images/eda/feature_correlation_heatmap.png'
]

RESULTS_IMAGE_FILES = [
    './images/results/random_forest_classifier_report.png',
    './images/results/logistic_regression_classifier_report.png',
    './images/results/feature_importance_plot.png',
    './images/results/feature_shap_explanation.png',
    './images/results/roc_curves_comparison.png'
]

MODEL_FILES = [
    './models/rfc_model.pkl',
    './models/logistic_model.pkl'
]

# Combined list for cleanup
ALL_TEST_FILES = EDA_IMAGE_FILES + RESULTS_IMAGE_FILES + MODEL_FILES


# Define the namedtuple (same as in churn_library.py)
ClassifierData = namedtuple('ClassifierData',
                            ['y_train',
                             'y_test',
                             'y_train_preds_lr',
                             'y_train_preds_rf',
                             'y_test_preds_lr',
                             'y_test_preds_rf'])


@pytest.fixture(scope="session", autouse=True)
def setup_session_logging():
    """
    Setup logging once for the entire test session
    """
    # Create logs directory
    Path('./logs').mkdir(exist_ok=True)

    # Create a timestamped log file for this test session
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f'./logs/churn_library_test_pytest_session_{timestamp}.log'

    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, mode='w'),
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )

    # Log session start
    session_logger = logging.getLogger('pytest_session')
    session_logger.info("=== Starting pytest test session ===")

    yield session_logger

    # Log session end
    session_logger.info("=== Ending pytest test session ===")

    # Cleanup
    logging.shutdown()

# Enhanced logging configuration for tests


@pytest.fixture(autouse=True)
def setup_test_logging(setup_session_logging):  # pylint: disable=redefined-outer-name, unused-argument
    # pylint seems not to know about fixtures and reports fixture use as W0621
    # and W0613
    """
    Setup logging for individual test execution
    """
    # Set up test logger
    test_logger = logging.getLogger('pytest_tests')
    test_logger.setLevel(logging.INFO)

    # Ensure churn library loggers also output to console
    churn_eda_logger = logging.getLogger('churn.eda')
    churn_lib_logger = logging.getLogger('churn.library')

    # Add console handlers if they don't exist
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    for logger in [churn_eda_logger, churn_lib_logger]:
        if not any(isinstance(h, logging.StreamHandler)
                   for h in logger.handlers):
            logger.addHandler(console_handler)

    yield test_logger


@pytest.fixture(autouse=True)
def test_separator(request):
    """Add separators between tests in log"""
    test_logger = logging.getLogger('pytest_tests')

    # Log test start
    test_logger.info("=" * 80)
    test_logger.info("STARTING TEST: %s", request.node.name)
    test_logger.info("=" * 80)

    yield

    # Log test end
    test_logger.info("COMPLETED TEST: %s", request.node.name)
    test_logger.info("-" * 80)


@pytest.fixture
def clean_images_dir():  # pylint: disable=redefined-outer-name
    """Fixture to clean up test images after test"""

    # Setup: ensure directories exist
    os.makedirs('./images', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)

    yield  # Run the test
    # Cleanup after test
    for file_path in EDA_IMAGE_FILES + RESULTS_IMAGE_FILES:
        if os.path.exists(file_path):
            os.remove(file_path)


@pytest.fixture
def clean_all_files():  # pylint: disable=redefined-outer-name
    """
    Fixture to clean up generated files before and after test
    """

    # Setup: ensure directories exist
    os.makedirs('./images/eda', exist_ok=True)
    os.makedirs('./images/results', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)

    for file_path in ALL_TEST_FILES:
        if os.path.exists(file_path):
            os.remove(file_path)

    yield  # Run the test
    # Cleanup after test is not done in order to keep results
    # for manual checking


@pytest.fixture
def sample_dataframe():  # pylint: disable=redefined-outer-name
    """
    Fixture to provide sample dataframe for testing
    """
    return cls.import_data("./data/bank_data.csv")


@pytest.fixture
def model_and_data(sample_dataframe):  # pylint: disable=redefined-outer-name
    """
    Fixture to provide trained models and data for testing
    """
    # prepare input data for classification report
    cls.perform_eda(sample_dataframe)
    x_train, x_test, y_train, y_test = cls.perform_feature_engineering(
        sample_dataframe, 'Churn')

    rfc = RandomForestClassifier(random_state=42)
    model_lrc = LogisticRegression(solver='liblinear', max_iter=3000)
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    model_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    model_rfc.fit(x_train, y_train)

    model_lrc.fit(x_train, y_train)
    y_train_preds_rf = model_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = model_rfc.best_estimator_.predict(x_test)

    y_train_preds_lr = model_lrc.predict(x_train)
    y_test_preds_lr = model_lrc.predict(x_test)

    classifier_data = ClassifierData(y_train=y_train,
                                     y_test=y_test,
                                     y_train_preds_lr=y_train_preds_lr,
                                     y_train_preds_rf=y_train_preds_rf,
                                     y_test_preds_lr=y_test_preds_lr,
                                     y_test_preds_rf=y_test_preds_rf
                                     )
    yield model_rfc, model_lrc, x_test, y_test, classifier_data


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
        test_logger.info(
            "Testing import_data: SUCCESS - caught FileNotFoundError")


def test_perform_eda(sample_dataframe, clean_images_dir):  # pylint: disable=redefined-outer-name, unused-argument
    # pylint seems not to know about fixtures and reports fixture use as W0621
    # and W0613
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


def test_encoder_helper(sample_dataframe):  # pylint: disable=redefined-outer-name
    # pylint seems not to know about fixtures and reports fixture use as W0621
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
        assert encoded_col in df_encoded.columns, f"Column {
            encoded_col} not found"
        assert df_encoded[encoded_col].dtype in ['float64', 'float32'], \
            f"Column {encoded_col} should be numeric"

    test_logger.info("Testing encoder_helper: SUCCESS")


def test_perform_feature_engineering(sample_dataframe):  # pylint: disable=redefined-outer-name
    # pylint seems not to know about fixtures and reports fixture use as W0621
    '''
    test perform_feature_engineering
    '''
    test_logger = logging.getLogger('pytest_tests')
    test_logger.debug("Testing perform_feature_engineering : START")
    # Test feature engineering
    # prepare data first:
    cls.perform_eda(sample_dataframe)
    x_train, x_test, y_train, y_test = cls.perform_feature_engineering(
        sample_dataframe, 'Churn')

    # Check shapes
    assert x_train.shape[0] > 0, "X_train is empty"
    assert x_test.shape[0] > 0, "X_test is empty"
    assert len(y_train) > 0, "y_train is empty"
    assert len(y_test) > 0, "y_test is empty"

    # Check that train and test sets have same number of features
    assert x_train.shape[1] == x_test.shape[1], "Train and test feature counts differ"

    # Check that split ratio is approximately correct (70/30)
    total_samples = len(y_train) + len(y_test)
    test_ratio = len(y_test) / total_samples
    assert 0.25 <= test_ratio <= 0.35, f"Test ratio {
        test_ratio} not around 0.3"

    test_logger.info("Testing perform_feature_engineering: SUCCESS")


def test_feature_importance_plot(sample_dataframe, clean_images_dir):  # pylint: disable=redefined-outer-name, unused-argument
    # pylint seems not to know about fixtures and reports fixture use as W0621
    # and W0613
    """
    test feature importance plotting
    """
    test_logger = logging.getLogger('pytest_tests')
    test_logger.debug("Testing classification_report_image : START")

    # prepare input data for classification report
    cls.perform_eda(sample_dataframe)
    x_train, x_test, y_train, _ = cls.perform_feature_engineering(
        sample_dataframe, 'Churn')

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


def test_classification_report_image(clean_images_dir,  # pylint: disable=redefined-outer-name, unused-argument
                                     model_and_data):  # pylint: disable=redefined-outer-name, unused-argument
    # pylint seems not to know about fixtures and reports fixture use as W0621
    # and W0613
    """
    test classification report image
    """
    test_logger = logging.getLogger('pytest_tests')
    test_logger.debug("Testing classification_report_image : START")

    # get prepared data from fixture
    _, _, _, _, classifier_data = model_and_data

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


def test_shap_explanation_plot(sample_dataframe, clean_images_dir):  # pylint: disable=redefined-outer-name, unused-argument
    # pylint seems not to know about fixtures and reports fixture use as W0621
    """
    test SHAP explanation plotting
    """
    test_logger = logging.getLogger('pytest_tests')
    test_logger.debug("Testing shap_explanation : START")

    # prepare input data for SHAP explanation
    cls.perform_eda(sample_dataframe)
    x_train, x_test, y_train, _ = cls.perform_feature_engineering(
        sample_dataframe, 'Churn')

    rfc = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)

    cls.shap_explanation_plot(
        cv_rfc.best_estimator_,
        x_test,
        './images/results/feature_shap_explanation.png')
    expected_files = [
        './images/results/feature_shap_explanation.png'
    ]
    for file_path in expected_files:
        assert os.path.exists(file_path), f"File {file_path} was not created"
        assert os.path.getsize(file_path) > 0, f"File {file_path} is empty"

    test_logger.info("Testing shap_explanation plotting: SUCCESS")


def test_roc_curve_report_image(sample_dataframe, clean_images_dir):  # pylint: disable=redefined-outer-name, unused-argument
    # pylint seems not to know about fixtures and reports fixture use as W0621
    """
    test ROC curve plotting
    """
    test_logger = logging.getLogger('pytest_tests')
    test_logger.debug("Testing roc_curve_plot : START")

    # prepare input data for ROC curve
    cls.perform_eda(sample_dataframe)

    x_train, x_test, y_train, y_test = cls.perform_feature_engineering(
        sample_dataframe, 'Churn')

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

    cls.roc_curve_report_image(cv_rfc, lrc, x_test, y_test)

    expected_files = [
        './images/results/roc_curves_comparison.png'
    ]
    for file_path in expected_files:
        assert os.path.exists(file_path), f"File {file_path} was not created"
        assert os.path.getsize(file_path) > 0, f"File {file_path} is empty"

    test_logger.info("Testing ROC curve plotting: SUCCESS")


def test_train_models(sample_dataframe):  # pylint: disable=redefined-outer-name
    # pylint seems not to know about fixtures and reports fixture use as W0621
    '''
    test train_models
    '''
    test_logger = logging.getLogger('pytest_tests')
    test_logger.debug("Testing train_models : START")
    # before we can test the training we have to prepare the data:
    cls.perform_eda(sample_dataframe)
    x_train, x_test, y_train, y_test = cls.perform_feature_engineering(
        sample_dataframe, 'Churn')

    # Train models
    cls.train_models(x_train, x_test, y_train, y_test)

    # Check that model files were created
    model_files = [
        './models/rfc_model.pkl',
        './models/logistic_model.pkl'
    ]

    for model_file in model_files:
        assert os.path.exists(model_file), f"Model file {
            model_file} not created"
        assert os.path.getsize(model_file) > 0, f"Model file {
            model_file} is empty"

    # Check that some evaluation images were created
    eval_images = [
        './images/logistic_regression_classifier_report.png',
        './images/random_forest_classifier_report.png'
    ]

    for image_file in eval_images:
        if os.path.exists(image_file):  # Some images might be optional
            assert os.path.getsize(image_file) > 0, f"Image file {
                image_file} is empty"

    test_logger.info("Testing train_models: SUCCESS")


def test_main(clean_all_files):  # pylint: disable=redefined-outer-name, unused-argument
    # pylint seems not to know about fixtures and reports fixture use as W0621
    """
    Test main function execution

    In order to make sure we do not accidentally have leftover files from
    previous test runs we use the clean_all_files fixture to clean up before
    running the main function.
    At the end we check that all expected output files were created and we
    leave them in place for manual checking.
    """
    test_logger = logging.getLogger('pytest_tests')
    test_logger.debug("Testing main function : START")

    try:
        cls.main()
        test_logger.info("Testing main function: SUCCESS")
    except Exception as err:
        test_logger.error("Testing main function: FAILED with error %s", err)
        raise err

    # Check that all expected output files were created and are non-empty
    for file_path in ALL_TEST_FILES:
        assert os.path.exists(file_path), f"File {file_path} was not created"
        assert os.path.getsize(file_path) > 0, f"File {file_path} is empty"

    test_logger.info("Testing main function: SUCCESS")
