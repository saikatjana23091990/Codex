#from json import dumps
from datetime import datetime
#import action_space
import pandas as pd
#import redis
#import ast
#import operator
#import re
#import json
import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt
#import plotly.graph_objs as go
import statsmodels.api as sm
#import math
#import statsmodels.api as smi
#import statsmodels.tsa.api as smt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import warnings
#warnings.filterwarnings("ignore")
#from sklearn.model_selection import train_test_split
#from sklearn.svm import SVC
#from sklearn.metrics import confusion_matrix
import time

def filter_and_update_dim_meas(dataframe,dimensions,measures,thres_for_dim=0.02,thres_for_meas=0.025):
    #measures=output['Measures']
    #dimensions=output['Dimensions']
    corr=dataframe.corr(method='pearson')
    #corr.shape

    newcorr=corr.loc[measures]
    otherCols=list(set(newcorr.columns).intersection(dimensions))
    newcorr=newcorr[otherCols]

    updated_measures=measures.copy()
    for col in measures:
        abs_sum=np.sum(np.abs(newcorr.loc[col]))
        avg_corr=abs_sum/len(otherCols)
        print(col,"=>>",avg_corr)
        if(avg_corr< thres_for_meas):
            #dimensions.append(col)
            updated_measures.remove(col)
            print("Removed")

    measures=updated_measures

    updated_dimensions=dimensions.copy()
    for col in otherCols:
        abs_sum=np.sum(np.abs(newcorr[col]))
        avg_corr=abs_sum/len(measures)
        print(col,"=>>",avg_corr)
        if(avg_corr<thres_for_dim):
            #dimensions.append(col)
            updated_dimensions.remove(col)
            print("Removed")

    dimensions=updated_dimensions

    return {'Dimensions':dimensions,'Measures':measures}

def backwardElimination(x, Y, sl, columns):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(Y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
                    columns = np.delete(columns, j)
                    
    regressor_OLS.summary()
    return x, columns

def get_dim_mes_keyColumns(df):
    #filename = 'uploadforanalysis_'+user+'.csv'
    #path = './adhocDataDirectory/'
    #df = pd.read_csv(path+filename)
    df = df.dropna(how='all')
    for col in df.columns:
        #df[col] = df[col].astype(str)
        if(df[col].dtype=='int64'):
            df[col] = df[col].fillna(0)
        elif(df[col].dtype=='float64'):
            df[col] = df[col].fillna(0.0)
        else:#(df[col].dtype=='object'):
            df[col] = df[col].fillna("Not Available")
            
        
    Dimensions = []
    Measures = []
    for var in df.columns:
        print(var)
        #print("UNIQUE",df[var].dropna().nunique(),"COUNT",df[var].dropna().count(),"FRACTION",df[var].dropna().nunique()/df[var].dropna().count())
        if ((1.*df[var].dropna().nunique()/df[var].dropna().count() < 0.05) == True):
            Dimensions.append(var);
            print(var,"added to Dimensions\n")
        else:
            if (df[var].dtypes != 'O'):
                Measures.append(var);print(var,"DATATYPE",df[var].dtypes," >added to Measures\n")
            else:
                Dimensions.append(var);print(var,"DATATYPE",df[var].dtypes,"added to Dimensions[else part]\n")
                
    
    df_dim = df[Dimensions]
    
    label_encoder = LabelEncoder()
    for i in range(len(df_dim.columns)):
        #print(df_dim.columns[i])
        #print(df_dim.iloc[:,i][0:2],end="\n")
        df_dim.iloc[:,i] = label_encoder.fit_transform(df_dim.iloc[:,i]).astype('float64')
        #print(df_dim.iloc[:,i][0:2],end="\n\n\n")
        #time.sleep(2)
    df_mes = df[Measures]
    df_clean = pd.concat([df_dim,df_mes],axis=1)
    
    corr = df_clean.corr()
    
    columns = np.full((corr.shape[0],), True, dtype=bool)
    for i in range(corr.shape[0]):
        for j in range(i+1, corr.shape[0]):
            if corr.iloc[i,j] >= 0.9:
                if columns[j]:
                    columns[j] = False
    selected_columns = df_clean.columns[columns]
    df_clean = df_clean[selected_columns]
    
    selected_columns = selected_columns[1:].values
    SL = 0.05
    data_modeled, selected_columns = backwardElimination(df_clean.iloc[:,1:].values, df_clean.iloc[:,0].values, SL, selected_columns)
    
    data_clean = pd.DataFrame(data = data_modeled, columns = selected_columns)
    
    
    imp_cat = []
    for cat in Dimensions:
        for col in data_clean.columns.values:
            if (cat == col):
                imp_cat.append(cat)

    
    DimwiseCount = []

    fetched_df = df

    for col in Dimensions:
        try:
            if fetched_df[col].dtypes == 'object':
                fetched_df[col] = pd.to_datetime(fetched_df[col])
        except:
            continue

    for catvar in Dimensions:
        DimwiseCount.append(catvar+'-'+str(df[catvar].nunique())+'-'+str(fetched_df[catvar].dtypes))
    
    returned_obj = {"Dimensions":Dimensions, "Measures":Measures, "Impactful features":imp_cat, "DimwiseCount":DimwiseCount}
    return returned_obj


