#!/usr/bin/env python3
""" module for converting column types """
import pandas as pd


def convert_columns(df):
    """ converts to numeric and to "No" and "Yes" """
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['SeniorCitizen'] = df['SeniorCitizen'].map({0: "No", 1: "Yes"})
    return df
