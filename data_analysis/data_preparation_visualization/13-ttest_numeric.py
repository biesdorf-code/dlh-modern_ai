#!/usr/bin/env python3
""" module for Welch's t-tests of numeric features vs churn """
from scipy import stats


def ttest_numeric(df):
    """ returns a dict of {feature: Welch's t-test p-value} vs Churn """
    results = {}
    features = df.select_dtypes(include="number").columns
    for col in features:
        yes = df[df['Churn'] == 'Yes'][col].dropna()
        no = df[df['Churn'] == 'No'][col].dropna()
        p_value = stats.ttest_ind(yes, no, equal_var=False)[1]
        results[col] = p_value
    return results
