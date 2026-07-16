#!/usr/bin/env python3
""" module for handling missing values in the TotalCharges column """


def clean_total_charges(df, method='drop'):
    """ 'drop' removes, 'median' fills with the column median, and
    'impute' replaces with MonthlyCharges * tenure.
    """
    df = df.copy()
    if method == 'drop':
        df = df.dropna(subset=['TotalCharges'])
    elif method == 'median':
        df['TotalCharges'] = df['TotalCharges'].fillna(
            df['TotalCharges'].median())
    elif method == 'impute':
        df['TotalCharges'] = df['TotalCharges'].fillna(
            df['MonthlyCharges'] * df['tenure'])
    return df
