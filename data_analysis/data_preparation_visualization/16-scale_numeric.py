#!/usr/bin/env python3
""" module for scaling numeric features """
from sklearn import preprocessing


def scale_numeric(df):
    """ standardizes MonthlyCharges and TotalCharges to mean 0, std 1 """
    cols = ['MonthlyCharges', 'TotalCharges']
    scaler = preprocessing.StandardScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df
