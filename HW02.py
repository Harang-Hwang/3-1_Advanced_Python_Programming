# -*- coding: utf-8 -*-
"""
Created on Mon May  5 14:16:54 2025

@author: haran
"""

# %%
#20231052_haranghwang

#Q1

import numpy as np

A = np.array([[2, 1, -1],
              [-3, -1, 2],
              [-2, 1, 2]])
b = np.array([8, -11, -3])

# 조건 1
x_solve = np.linalg.solve(A, b)
print("Q1_조건 1 | solve()로 구한 해:", x_solve)
print()

# 조건 2
A_inv = np.linalg.inv(A)
x_inv = A_inv @ b
print("Q1_조건 2 | inv()로 구한 해:", x_inv)
print()

# 조건 3
b_check = A @ x_solve
print("Q1_조건 3 | Ax의 결과:", b_check)
print("Q1_조건 3 | 원래 b와 같은가?:", np.allclose(b, b_check))
print()
print()


# %%
#20231052_haranghwang

#Q2

import joblib

arr1 = np.random.randint(0, 10, (3, 4))
arr2 = np.random.randint(10, 20, (3, 4))

# 조건 1
np.save('array1.npy', arr1)
np.savez('array2.npz', arr2=arr2)
joblib.dump(arr1, 'array3.pkl')

# 조건 2
arr1_loaded = np.load('array1.npy')
print("Q2_조건 2 | array1.npy 불러오기:\n", arr1_loaded)
print()

arr2_loaded = np.load('array2.npz')['arr2']
print("Q2_조건 2 | array2.npz 불러오기:\n", arr2_loaded)
print()

arr3_loaded = joblib.load('array3.pkl')
print("Q2_조건 2 | array3.pkl 불러오기:\n", arr3_loaded)
print()
print()


# %%
#20231052_haranghwang

import pandas as pd

glucose_data = {
    'subject_01': 98,
    'subject_02': 110,
    'subject_03': 87,
    'subject_04': 145,
    'subject_05': 132,
    'subject_06': 120,
}

# 조건 1
glucose = pd.Series(glucose_data)

# 조건 2
mean_glucose = np.mean(glucose)
print("Q3_조건 2 | 전체 평균 혈당 수치:", mean_glucose)
print()

# 조건 3
high_glucose = glucose[glucose >= 125]
print("Q3_조건 3 | 혈당 수치 125 이상:\n", high_glucose)
print()

# 조건 4
subset = glucose['subject_02':'subject_05']
print("Q3_조건 4 | subject_02 ~ subject_05:\n", subset)