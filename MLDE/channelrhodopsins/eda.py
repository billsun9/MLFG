# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 19:08:51 2023

@author: Bill Sun
"""

import pandas as pd
import numpy as np

PATH = './Data/ChR/pnas.1700269114.sd01.csv'

df = pd.read_csv(PATH)
print(df.columns)
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def check(df):
    badIdxs = set()
    for i in range(len(df)):
        ex = df.iloc[i]
        if len(set([s.lower() for s in ex['sequence']])) != 4:
            print(set(ex['sequence']))
        
        if not is_number(ex['mKate_mean']):
            print(i, ex['mKate_mean'])
            badIdxs.add(i)
        if not is_number(ex['GFP_mean']):
            print(i, ex['GFP_mean'])
            badIdxs.add(i)
        if not is_number(ex['intensity_ratio_mean']):
            print(i, ex['intensity_ratio_mean'])
            badIdxs.add(i)
    return badIdxs

def clean(df):
    print(["Before Dropping"])
    badIdxs = check(df)
    df = df.drop(labels=badIdxs)
    df['sequence'] = df['sequence'].apply(lambda x: x.upper())
    df = df.reset_index(drop=True)
    df = df.rename(columns={"sequence":"Sequence", "GFP_mean": "Data"}, errors="raise")
    df['Sequence'].astype(str)
    df['Data'].astype(float)
    # print(df.columns)
    return df[['Sequence','Data']]

df = clean(df)
print(df.columns)