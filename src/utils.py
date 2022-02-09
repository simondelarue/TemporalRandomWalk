# -*- coding: utf-8 -*-

import pandas as pd

def switch_cols_df(df, c1, c2) -> pd.DataFrame :
    ''' Switch columns in DataFrame.
        
        Parameters
        ----------
            df : DataFrame
            c1, c2 : Labels of columns in df to switch. 
            
        Output
        ------
            DataFrame with switched columns. '''

    sw_df = df.copy()
    cols = list(df.columns)
    a, b = cols.index(c1), cols.index(c2)
    cols[b], cols[a] = cols[a], cols[b]

    sw_df.columns = cols

    return  sw_df