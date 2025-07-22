# -*- coding: utf-8 -*-
"""
Created on Wed May 14 22:33:08 2025

@author: haran
"""

# %%
#20231052_haranghwang

#Q1

import pandas as pd
import numpy as np

data_dict = {
    'name': ['Wonyoung', 'Soyeon', 'Jisoo', 'Hyori'],
    'group': ['IVE', 'GIdle', 'BlackPink', 'FinKL'],
    'age': [22, 28, 31, 47],
    'MBTI': ['ENTJ', 'INTP', 'INTP', 'ENFP']
}

df1 = pd.DataFrame(data_dict)

print("[Q1]")
print("리스트 기반 딕셔너리 DataFrame:")
print(df1)
print('='*50)


dtype = np.dtype([('name', 'U10'), ('group', 'U10'), ('age', 'i4'), ('MBTI', 'U4')])
data_struct = np.array([('Wonyoung', 'IVE', 22, 'ENTJ'),
                        ('Soyeon', 'GIdle', 28, 'INTP'),
                        ('Jisoo', 'BlackPink', 31, 'INTP'),
                        ('Hyori', 'FinKL', 47, 'ENFP')], dtype=dtype)

df2 = pd.DataFrame(data_struct)
print("구조화된 배열 기반 DataFrame:")
print(df2)
print()
print()

# %%
#20231052_haranghwang

#Q2

# 조건1
print("[Q2]")
print("조건1: 나이가 30 이상")
print(df1[df1['age'] >= 30])
print('='*50)

# 조건2
print("조건2: MBTI가 'INTP'")
print(df1[df1['MBTI'] == 'INTP'])
print('='*50)

# 조건3
print("조건3: 첫 번째 ~ 세 번째 행")
print(df1.iloc[0:3])  
print()
print()

# %%
#20231052_haranghwang

#Q3

data = {
    'ID': [1, 2, 3, 4],
    'Score': [88, 92, 75, 95],
    'Grade': ['B', 'A', 'C', 'A']
}

# 조건1
index_labels = ['s1', 's2', 's3', 's4']
df = pd.DataFrame(data, index=index_labels)

print("[Q3]")
print("조건1: 고유 인덱스 DataFrame")
print(df)
print('='*50)

# 조건2: 새로운 학생 s5 추가
df.loc['s5'] = [5, 80, 'B']
print("조건2: s5 행 추가")
print(df)
print('='*50)

# 조건3: 'Score' 열 삭제
del df['Score']
print("조건3: 'Score' 열 삭제")
print(df)
print('='*50)

# 조건4: 's3', 's4' 행 삭제 + 'Grade' 열 삭제
df = df.drop(['s3', 's4'])  # 행 삭제
df = df.drop(['Grade'], axis=1)  # 열 삭제
print("조건4: s3, s4 제거 후 'Grade' 열 삭제")
print(df)
