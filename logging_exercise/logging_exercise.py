""" This is exercise 4.19 from Udacity Training Machine Learning DevOps Engineer Nanodegree.
The goal of this exercise is to practice logging and exception handling."""
## STEPS TO COMPLETE ##
# 1. import logging
# 2. set up config file for logging called `results.log`
# 3. add try except with logging for success or error
#    in relation to checking the types of a and b
# 4. check to see that log is created and populated correctly
#    should have error for first function and success for
#    the second
# 5. use pylint and autopep8 to make changes
#    and receive 10/10 pep8 score
import logging

logger = logging.getLogger(__name__)


def config_logger():
    """Configures logging to log to a file named results.log"""
    logging.basicConfig(filename="./results.log", level=logging.INFO)


def sum_vals(a, b):
    """Returns the sum of a and b including some error checking
    
    Args:
        a: (int)
        b: (int)
    Return:
        a + b (int)
    """

    error = False
    if not isinstance(a, (int, float)):
        logging.error("Parameter a is not a number: %s.", a)
        error = True
    if not isinstance(b, (int, float)):
        logging.error("Parameter b is not a number: %s.", b)
        error = True
    if error is False:
        c = a + b
        logging.info("SUCCESS: Computed a+b = %s.", c)
        return c
    #if we get here, there was an error
    return "a + b cannot be computed"


if __name__ == "__main__":
    config_logger()
    sum_vals('no', 'way')
    sum_vals(4, 5)
