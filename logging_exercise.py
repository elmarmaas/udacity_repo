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
    logging.basicConfig(filename="./results.log", level=logging.INFO)
    
def sum_vals(a, b):
    '''
    Args:
        a: (int)
        b: (int)
    Return:
        a + b (int)
    '''
    error = False
    if not (isinstance(a, (int, float))):
        logging.error(f"Parameter a is not a number: {a}.")
        error = True
    if not (isinstance(b, (int, float))):
        logging.error(f"Parameter b is not a number: {b}.")
        error = True
    if (error == False):
        c = a+b
        logging.info(f"SUCCESS: Computed a+b = {c}.")
        return c

if __name__ == "__main__":
    config_logger()
    sum_vals('no', 'way')
    sum_vals(4, 5)
