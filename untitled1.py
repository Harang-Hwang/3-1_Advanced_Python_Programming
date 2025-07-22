# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 11:25:24 2025

@author: haran
"""

import numpy as np

a = np.array([1,2,3])

print(f'a:{a}')

b = np.arange(5)

print(f'b:{np.array(b)}')

x=[1,2,3]
y=[[1,2,3], [2,3,4], [3,4,5]]

arr1 = np.array(x)
arr2 = np.array(y)

print(arr1)
print(arr2)

print(f'arr1+arr2 \n = {arr1+arr2}')

