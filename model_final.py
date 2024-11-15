import pandas as pd  
import numpy as np
from skimpy import skim

import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, precision_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.feature_selection import SelectKBest, f_classif

from xgboost import XGBClassifier

from multiprocessing import pool

from keras.models import Sequential
from keras.layers import Dense, Dropout, Normalization
from keras.losses import BinaryCrossentropy

import optuna
import pickle
import os

def preprocess_data(master_df):
    
    drop_enc = OneHotEncoder(drop='first').fit(pd.DataFrame(master_df['carrier']))
    new_values = drop_enc.transform(pd.DataFrame(master_df['carrier'])).toarray()

    df = master_df.drop('carrier',axis=1)
    df[drop_enc.get_feature_names_out()[0]] = new_values[:,0]
    df[drop_enc.get_feature_names_out()[1]] = new_values[:,1]

    df['package_type'] = df['package_type'].apply(encode_size)
    df['shipping_method'] = df['shipping_method'].apply(encode_shipping)
    df['holiday_period'] = df['holiday_period'].apply(encode_holiday)
    df['customer_tier'] = df['customer_tier'].apply(encode_customer)
    df['weather_conditions'] = df['weather_conditions'].apply(encode_weather)
    df['pop_density'] = df['pop_density'].apply(encode_population)
    df['location'] = df['location'].apply(encode_town)

    return df
    
def run_Production_Model(data):
    
    try: 
        with open("final_model.pkl", "rb") as f:
            loaded_model = pickle.load(f)
            
        cols_df = pd.read_csv('col_df')
        columns = cols_df['columns']
        data_final = StandardScaler().fit_transform(data[columns])
        
        preds = loaded_model.predict(data_final)

        print('Modeling successful!')
        return preds
            
    except:
        print('Modeling failed.')

def export_model_and_cols(model, cols):
    with open('final_model.pkl', 'wb') as file:
        pickle.dump(model, file)
        
    col_df = pd.DataFrame({
        'idx': range(len(cols)),
        'columns': cols
    })
    
    col_df.to_csv('col_df',index=False)
    
def encode_size(x):
    
    size_dict = {
        'Medium':0,
        'Large':1
    }
    
    return size_dict[x]

def encode_shipping(x):
    
    size_dict = {
        'Expedited':0,
        'Standard':1
    }
    
    return size_dict[x]

def encode_holiday(x):
    
    size_dict = {
        'Holiday_Period_No':0,
        'Holiday_Period_Yes':1
    }
    
    return size_dict[x]

def encode_weather(x):
    
    size_dict = {
        'Clear':0,
        'Rain':1,
        'Snow':2
    }
    
    return size_dict[x]

def encode_population(x):
    
    size_dict = {
        'Low':0,
        'Medium':1,
        'High':2
    }
    
    return size_dict[x]

def encode_customer(x):
    
    size_dict = {
        'Basic':1,
        'Premium':0
    }
    
    return size_dict[x]

def encode_town(x):
    
    size_dict = {
        'Rural':0,
        'Suburban':1,
        'Urban':2
    }
    
    return size_dict[x]


if __name__ == '__main__':
    df = pd.read_csv('classification_test.csv')
    df_preprocessed = preprocess_data(df)
    preds = run_Production_Model(df_preprocessed)

    df['shipment_issue'] = preds
    df.to_csv('out_of_sample_predictions', index=False)

