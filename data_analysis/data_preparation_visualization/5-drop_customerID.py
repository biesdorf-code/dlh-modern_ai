#!/usr/bin/env python3
""" module for dropping the customerID from a df """
import pandas as pd


def drop_customerID(df):
    """ drops the customerID column """
    return df.drop(columns=['customerID'])
