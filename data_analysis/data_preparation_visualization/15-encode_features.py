#!/usr/bin/env python3
""" module for encoding features for modeling """
import pandas as pd
from sklearn import preprocessing


def encode_features(df):
    """ encodes features and returns df plus fitted encoders """
    df = df.copy()

    # Churn: LabelEncoder (No->0, Yes->1)
    churn_le = preprocessing.LabelEncoder()
    # fit_transform produces a 1D numpy array of integers
    df['Churn'] = churn_le.fit_transform(df['Churn'])

    # binary columns: OrdinalEncoder (No->0, Yes->1)
    binary_cols = ['Partner', 'Dependents', 'PaperlessBilling',
                   'SeniorCitizen']
    binary_oe = preprocessing.OrdinalEncoder(
        categories=[['No', 'Yes']] * len(binary_cols))
    # fit_transform produces a 2D float array -> astype(int) converts to int64
    df[binary_cols] = binary_oe.fit_transform(
        df[binary_cols]).astype(int)

    # TenureGroup: OrdinalEncoder in alphabetical order
    tenure_oe = preprocessing.OrdinalEncoder()
    # double brackets produce a 2D DataFrame (OrdinalEncoder requires 2D input)
    df['TenureGroup'] = tenure_oe.fit_transform(
        df[['TenureGroup']].astype(str)).astype(int)

    # Contract, PaymentMethod: one-hot with drop_first
    # get_dummies furnishes a new DataFrame with the dummy columns appended
    df = pd.get_dummies(
        df, columns=['Contract', 'PaymentMethod'],
        drop_first=True, dtype=int)

    return df, churn_le, binary_oe, tenure_oe
