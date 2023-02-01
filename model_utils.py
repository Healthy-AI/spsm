import pandas as pd
import numpy as np

def get_array(x):
    """ Returns x if x is an array and the values of x if x is a dataframe 
    
    Args: 
        x (DataFrame, Series or ndarray)
        
    Returns: 
        ndarray: Numpy array matching the input
    """
    
    if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
        return x.values
    elif isinstance(x, np.ndarray):
        return x
    else:
        raise Exception('Unknown data format of input')
        
def get_dataframe(x):
    """ Returns x if x is an dataframe and a dataframe of x if x is an array 
    
    Args: 
        x (DataFrame, Series or ndarray)
        
    Returns: 
        DataFrame: Pandas DataFrame matching the input
    """
    
    if isinstance(x, pd.DataFrame):
        return x
    elif isinstance(x, np.ndarray):
        return pd.DataFrame(x)
    else:
        raise Exception('Unknown data format of input')   