# %%

# Import libraries
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from scipy.stats import linregress

def scores_df(file_dir,csv_name,header,ColumnName1,ColumnName2):



    # %%
    # Load patient data
    os.chdir(file_dir)
    df = pd.read_csv(csv_name)

    # %%
    # Make a copy
    dfd = df.copy()

    # %%
    # Drop blank columns
    for (columnName, columnData) in dfd.iteritems():
        if columnData.isnull().all():
            print('Dropping NaN column at',columnName)
            dfd.drop(columnName,axis=1,inplace=True)

    # %%
    # Add relevant column names from headers
    df_headers = []
    for (columnName, columnData) in dfd.iteritems():
        if 'Unnamed' not in columnName:
            df_headers.append(columnName)
        else:
            print('Renaming',columnName,'as',df_headers[-1]+' '+str(dfd.iloc[0, df.columns.get_loc(columnName)-1]))
            dfd.rename(columns={columnName:df_headers[-1]+' '+str(dfd.iloc[0, df.columns.get_loc(columnName)-1])},inplace=True)

    # %%
    # Make a copy for motor symptoms
    df_out = dfd.copy()
    # Drop non-motor (III) columns
    for (columnName, columnData) in dfd.iteritems():
        if header in columnName:
            next
        elif 'Anonymous ID' in columnName:
            df_out.iloc[0,0] = 'Anonymous ID'
        else:
            df_out.drop(columnName,axis=1,inplace=True)

    # %%
    # Rename columns with specific metrics
    df_out.columns = df_out.iloc[0]
    df_out = df_out.tail(-1)


    # %%
    # Convert columns to numerical arrays
    score1 = df_out[ColumnName1].to_numpy().astype('float')
    score2 = df_out[ColumnName2].to_numpy().astype('float')
    

    # %%
    # Find numerical entries only
    cases = []
    for ids in np.arange(0,score1.__len__()):
        if ~np.isnan(score1[ids]) and ~np.isnan(score2[ids]): 
            cases.append(ids)

    # %%
    anon_ids = (dfd['Anonymous ID'].tail(-1)).to_numpy().astype('float')


    return df_out,anon_ids,cases