#!/usr/bin/env python3
""" module for engineering new features from the dataset """
import pandas as pd


def create_features(df):
    """ creates NumServices and TenureGroup, drops source columns """
    service_cols = [
        'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies'
    ]

    # count 'Yes' across telecom service labels
    df['NumServices'] = (df[service_cols] == 'Yes').sum(axis=1)
    # add internet subscription
    df['NumServices'] += (df['InternetService'] != 'No').astype(int)

    # bin limits: 0 excluded, upper bounds included
    df['TenureGroup'] = pd.cut(
        df['tenure'],
        bins=[0, 12, 24, 48, 60, float('inf')],
        labels=['0-12', '13-24', '25-48', '49-60', '60+']
    )

    df = df.drop(columns=service_cols + ['InternetService', 'tenure'])
    return df
