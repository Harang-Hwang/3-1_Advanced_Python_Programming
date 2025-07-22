# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 12:02:44 2025

@author: haran
"""

# %%
# 20231052_HarangHwang

import numpy as np

X = np.random.randn(10,4)

print(X)

u = np.mean(X)

print(u)

a = np.std(X)

print(a)

Z = (X-u)/a

S = 1/(n-1)*Z@Z


# %%
# 20231052_HarangHwang

dt = {("names:{'id', 'height', 'weight', 'age'}"), ("formats: {'i4', 'f4', 'f4', 'i4'}")}

arr0 = np.array([('101', '102', '103', '104'), ('1.85', '1.70', '1.60', '1.75'), ('85.0', '67.5', '80.5', '65.0'), ('25', '32', '45', '29')])

BMI = arr0[weight] / arr0[height]**2

print(BMI)

print(arr0[id], 'BMI >= 23')


# %%
# 20231052_HarangHwang

arr1 = np.arange(9).reshape(3,3)

arr2 = np.arange(9).reshape(3,3)

sub_arr1 = arr1[1:,1:]

sub_arr2 = arr2[:2, :2]

sub_arr1[1,1] = 100

sub_arr2[0,0] = 10

print(sub_arr1) # 변경됨, 이유 : 슬라이싱은 원본 배열도 바뀜
print(sub_arr2) # 변경되면 안되는데 코딩 잘못해서 변경됨, 이유 : 카피했으므로 원본 배열은 바뀌면 안됨

arr3 = np.vstack((sub_arr1, sub_arr2))





# %%
# 20231052_HarangHwang

A = np.random.normal(2, 3, 100)

a = np.mean(A)
b = np.std(A)

print(a)
print(b)


A_out = a+1.5*b < int(x) < a-1.5*b

print(A_out)

A_normal = np.subtract(A, A_out)

print(A_normal)
print(np.size(A_normal))


# %%
# 20231052_HarangHwang

arr_x = np.linspace(10,40,15)


arr_y = arr_x.reshape(5,3)

arr_z = arr_y[[0,2], [1,1], [2,1], [3,1], [4,0]]

print(arr_z)


# %%
# 20231052_HarangHwang

dt = ('f8')

arr_c = np.arange(5, 29, 1).reshape(2,3,4).dtype=dt

arr_c = arr_c.ravel(order = 'C')

arr_f = arr_c.ravel(order = 'F')

print(np.strides(arr_c))
print(np.strides(arr_f))

# stride가 그렇게 나오는 이유 : arr_c의 경우, C 우선 방식으로 행 우선 방식으로 메모리 상에 배치했기 때문이다. arr_f의 경우, F 우선 방식으로 열 우선 방식으로 메모리 상에 배치했기 때문이다.

print(np.flatten(arr_c))
print(np.flatten(arr_f))







