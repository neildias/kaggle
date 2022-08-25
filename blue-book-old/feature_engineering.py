import numpy as np
import pandas as pd
import re

def numericalise(old_df):
    """Numericalises all categorical columns"""
    df = old_df.copy()

    for col in df.select_dtypes(include='O').columns:
        df[col] = df[col].astype("category").cat.codes
    return df

def date_feature_extracter(df,
                            colnames,
                            drop=True,
                            time=False,
                            errors='raise'):
    """Extracts datas from a given dataframe"""
    if isinstance(colnames,str): 
        colnames = [colnames]    #convert single string to list
    for colname in colnames:
        col = df[colname]
        col_dtype = col.dtype
        # get time dtype
        if isinstance(col_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
            col_dtype = np.datetime64
        #dt transformation
        if not np.issubdtype(col_dtype, np.datetime64):
            df[colname] = col = pd.to_datetime(col, 
                                               infer_datetime_format=True, 
                                               errors=errors)
        targ_pre = re.sub('[Dd]ate$', '', colname)

        #extract various date attributes of pandas
        attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
                'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
        if time: attr = attr + ['Hour', 'Minute', 'Second']
        for n in attr: df[targ_pre + n] = getattr(col.dt, n.lower())
        df[targ_pre + 'Elapsed'] = col.astype(np.int64) // 10 ** 9
        if drop: df.drop(colname, axis=1, inplace=True)