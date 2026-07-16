#!/usr/bin/env python3
""" module for removing duplicate rows from a df """


def remove_duplicates(df):
    """ drops all duplicate rows """
    return df.drop_duplicates()
