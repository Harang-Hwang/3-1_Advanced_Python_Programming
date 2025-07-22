# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 12:00:48 2025

@author: haran
"""

a = int(input ('나이 : '))

if 0<=a<=12:
    print ('알파세대')
elif 13<=a<=28:
    print('Z 세대')
elif 29<=a<=44:
    print('Y 세대')
elif 45<=a:
    print('X 세대')
else:
    print('0 이상의 숫자만 입력하세요.')

