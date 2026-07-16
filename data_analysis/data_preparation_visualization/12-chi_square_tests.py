#!/usr/bin/env python3
""" module for chi-square tests of categorical features vs churn """
import pandas as pd
from scipy import stats


def chi_square_tests(df):
    """ returns a dict of {feature: chi-square p-value} vs Churn """
    results = {}
    features = df.select_dtypes(include="object").columns
    for col in features:
        if col == 'Churn':
            continue
        contingency = pd.crosstab(df[col], df['Churn'])
        p_value = stats.chi2_contingency(contingency)[1]
        results[col] = p_value
    return results
