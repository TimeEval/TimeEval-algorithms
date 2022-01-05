import numpy as np

def preprocess(df):
    
    if len(df.shape) == 1:
        raise ValueError('Data must be a 2-D array')

    if np.any(np.isnan(df)):
        print('Data contains null values. Will be replaced with 0')
        df = np.nan_to_num(df)

    print('Data is clean')
    return df