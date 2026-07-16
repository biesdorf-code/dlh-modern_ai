#!/usr/bin/env python3
""" module for dropping the customerID from a df """


def drop_customerID(df):
    """ drops the customerID column """
    return df.drop(columns=['customerID'])
