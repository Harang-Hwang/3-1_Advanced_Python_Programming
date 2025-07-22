# -*- coding: utf-8 -*-
"""
Created on Sat May 24 09:27:36 2025

@author: haran
"""

# %%
#20231052_haranghwang

#Q1

import numpy as np
import pandas as pd


df = pd.DataFrame({
    'ID': ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7'],
    'BP': [np.nan, 150, 120, np.nan, 130, 135, np.nan]
})

# 1
bp_mean = df['BP'].mean()
print("[Q1]")
print("1) 평균 혈압 (결측치 제외):", bp_mean)
print()

# 2
df['BP_filled'] = df['BP'].fillna(bp_mean)
print("\n2) BP_filled 열 추가:\n", df)
print()

# 3
df['BP_Level'] = df['BP_filled'].apply(lambda x: 'High' if x > bp_mean else 'Normal')
print("\n3) BP_Level 열 추가:\n", df)
print()


# %%
#20231052_haranghwang

#Q2

import numpy as np
import pandas as pd

index = pd.MultiIndex.from_product(
    [['Asan', 'Yonsei'], pd.date_range('2025-05-23', periods=5)],
    names=['Hospital', 'Date']
)

temp = [np.nan, 36.5, 38.0, np.nan, 37.5, np.nan, 38.0, 38.5, 38.5, np.nan]
df = pd.DataFrame({'Temp': temp}, index=index)
print("[Q1]")
print("원본 데이터:\n", df)
print()

# 1
df_ffill = df.groupby(level='Hospital').ffill()
print("\n1) 병원별 전일값으로 결측치 채운 결과:\n", df_ffill)
print()

# 2
hospital_mean = df.groupby(level='Hospital')['Temp'].transform('mean')
print("\n2) 병원별 평균 온도:\n", hospital_mean)
print()

# 3
df['Temp_filled'] = df_ffill['Temp']
mask = df['Temp_filled'].isna()
df.loc[mask, 'Temp_filled'] = hospital_mean[mask]
print("\n3) Temp_filled 열 추가:\n", df)
print()

