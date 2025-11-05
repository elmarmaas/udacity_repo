'''
Module churn_script_logging_and_tests.py
Test script for all functions in churn_library.py
'''

import os
import logging
import pytest
from pathlib import Path
# import churn_library_solution as cls
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library_test.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


@pytest.fixture
def clean_images_dir():
    """Fixture to clean up test images after test"""
    yield  # Run the test
    # Cleanup after test
    image_files = [
        './images/churn_histogram.png',
        './images/customer_age_histogram.png',
        # ... other files
    ]
    for file_path in image_files:
        if os.path.exists(file_path):
            os.remove(file_path)


def test_import(import_data):
    '''
    test data import - test that data is loaded correctly
        and we get the expected number of rows and columns
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found")
        raise err
    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
        logging.info("Testing import_data - shape test: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err

    try:
        logging.info(
            "Testing import_data with nonexistent file... \
                - the following error is expected")
        # This should raise FileNotFoundError
        df = import_data("nonexistent_file.csv")
        logging.error(
            "Testing import_data: FAILED - No exception for nonexistent file")
    except FileNotFoundError as err:
        logging.info("Testing import_data: SUCCESS - caught FileNotFoundError")


def test_eda(perform_eda, clean_images_dir):
    '''
    test perform eda function with cleanup
    '''
    df = cls.import_data("./data/bank_data.csv")
    perform_eda(df)

    # Check that EDA images were created
    expected_files = [
        './images/churn_histogram.png',
        './images/customer_age_histogram.png',
        './images/customer_marital_status_historgram.png',
        './images/total_trans_ct_histogram.png',
        './images/feature_correlation_heatmap.png'
    ]

    for file_path in expected_files:
        assert os.path.exists(file_path), f"File {file_path} was not created"
        assert os.path.getsize(file_path) > 0, f"File {file_path} is empty"


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''


def test_train_models(train_models):
    '''
    test train_models
    '''


if __name__ == "__main__":
    test_import(cls.import_data)
    test_eda(cls.perform_eda, clean_images_dir)
