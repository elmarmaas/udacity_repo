
""" Exercise 4.22 from Udacity Training Machine Learning DevOps Engineer Nanodegree.
The goal of this exercise is to practice logging and exception handling."""
## STEPS TO COMPLETE ##
# 1. import logging
# 2. set up config file for logging called `test_results.log`
# 3. add try except with logging and assert tests for each function
#    - consider denominator not zero (divide_vals)
#    - consider that values must be floats (divide_vals)
#    - consider text must be string (num_words)
# 4. check to see that the log is created and populated correctly
#    should have error for first function and success for
#    the second
# 5. use pylint and autopep8 to make changes
#    and receive 10/10 pep8 score

import logging

logger = logging.getLogger(__name__)


def config_logger():
    """Configures logging to log to a file named test_results.log"""
    logging.basicConfig(filename="./test_results.log", level=logging.INFO)


def divide_vals(numerator, denominator):
    '''
    Args:
        numerator: (float) numerator of fraction
        denominator: (float) denominator of fraction

    Returns:
        fraction_val: (float) numerator/denominator
    '''
    try:
        fraction_val = numerator / denominator
        logger.info("SUCCESS: Computed numerator/denominator = %s.",
                    fraction_val)
        return fraction_val
    except ZeroDivisionError:
        logger.error("Denominator cannot be zero: %s.", denominator)
        return "denominator cannot be zero"


def num_words(text):
    '''
    Args:
        text: (string) string of words

    Returns:
        num_words: (int) number of words in the string
    '''
    try:
        words = len(text.split())
        logger.info("SUCCESS: Computed number of words = %s.", words)
        return words
    except AttributeError:
        logger.error("Text argument must be a string: %s.", text)
        return "text argument must be a string"


if __name__ == "__main__":
    config_logger()
    divide_vals(3.4, 0)
    divide_vals(4.5, 2.7)
    divide_vals(-3.8, 2.1)
    divide_vals(1, 2)
    num_words(5)
    num_words('This is the best string')
    num_words('one')
