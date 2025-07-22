### 13주차 강의

## 판다스 고급

### 데이터 가공


```python
import numpy as np
import pandas as pd
```


```python
# 1. 데이터 가공

# 데이터 이어 붙이기(pd.concat())
df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                    'B': ['B0', 'B1', 'B2'],
                    'C': ['C0', 'C1', 'C2']}, index=[0, 1, 2])
df2 = pd.DataFrame({'A': ['A3', 'A4', 'A5'],
                    'B': ['B3', 'B4', 'B5'],
                    'C': ['C3', 'C4', 'C5']}, index=[3, 4, 5])
df3 = pd.DataFrame({'A': ['A6', 'A7', 'A8'],
                    'B': ['B6', 'B7', 'B8'],
                    'C': ['C6', 'C7', 'C8']}, index=[6, 7, 8])

frames = [df1, df2, df3] # 연결할 객체들이 리스트에 담겨있음

# axis 매개변수가 지정되지 않았으므로 기본값인 axis=0
# (axis=0 -> 행을 기준으로 위아래로 이어 붙이기)
result = pd.concat(frames)
print(result), print('='*50)

# 합쳐진 결과 DataFrame에 새로운 인덱스 레벨을 추가
result1 = pd.concat(frames, keys=['x', 'y', 'z']) # 멀티 인덱스
print(result1), print('='*50)

print(result1.loc['z'])
```

        A   B   C
    0  A0  B0  C0
    1  A1  B1  C1
    2  A2  B2  C2
    3  A3  B3  C3
    4  A4  B4  C4
    5  A5  B5  C5
    6  A6  B6  C6
    7  A7  B7  C7
    8  A8  B8  C8
    ==================================================
          A   B   C
    x 0  A0  B0  C0
      1  A1  B1  C1
      2  A2  B2  C2
    y 3  A3  B3  C3
      4  A4  B4  C4
      5  A5  B5  C5
    z 6  A6  B6  C6
      7  A7  B7  C7
      8  A8  B8  C8
    ==================================================
        A   B   C
    6  A6  B6  C6
    7  A7  B7  C7
    8  A8  B8  C8
    


```python
# 1. 데이터 가공

# 데이터 이어 붙이기(pd.concat()): 축의 로직 설정으로 이어 붙이기
df4 = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                    'B': ['B0', 'B1', 'B2'],
                    'C': ['C0', 'C1', 'C2']}, index=[0, 2, 3])

df5 = pd.DataFrame({'B': ['B2', 'B6', 'B7'],
                    'C': ['C2', 'C6', 'C7'],
                    'E': ['E2', 'E6', 'E7']}, index=[2, 6, 7])

print(df4), print('='*50)
print(df5), print('='*50)

# axis=1: 데이터를 열(column)을 기준으로 결합. 즉, 두 옆으로 이어 붙임
# join='outer' (default)
# sort=False: 원래 DataFrame의 열 순서로 정렬하지 않고 유지
# (axis=1 -> 인덱스를 기준으로 좌우로 이어 붙이기)
# 인덱스가 일치하지 않는 부분은 NaN으로 채워짐
result = pd.concat([df4, df5], axis=1, sort=False)
#result = pd.concat([df4, df5], axis=1, sort=True)
print(result), print('='*50)

# join='inner': DataFrame들에 존재하는 인덱스(또는 열)의 교집합만 결과에 포함시킴
result1 = pd.concat([df4, df5], axis=1, join='inner')
print(result1), print('='*50)

# join='outer' (기본값): DataFrame들에 존재하는 인덱스(또는 열)의 합집합을 결과에 포함시킴
# 일치하지 않는 부분은 NaN으로 채움
result2 = pd.concat([df4, df5], axis=1, join='outer')
print(result2)
```

        A   B   C
    0  A0  B0  C0
    2  A1  B1  C1
    3  A2  B2  C2
    ==================================================
        B   C   E
    2  B2  C2  E2
    6  B6  C6  E6
    7  B7  C7  E7
    ==================================================
         A    B    C    B    C    E
    0   A0   B0   C0  NaN  NaN  NaN
    2   A1   B1   C1   B2   C2   E2
    3   A2   B2   C2  NaN  NaN  NaN
    6  NaN  NaN  NaN   B6   C6   E6
    7  NaN  NaN  NaN   B7   C7   E7
    ==================================================
        A   B   C   B   C   E
    2  A1  B1  C1  B2  C2  E2
    ==================================================
         A    B    C    B    C    E
    0   A0   B0   C0  NaN  NaN  NaN
    2   A1   B1   C1   B2   C2   E2
    3   A2   B2   C2  NaN  NaN  NaN
    6  NaN  NaN  NaN   B6   C6   E6
    7  NaN  NaN  NaN   B7   C7   E7
    


```python
# 1. 데이터 가공

# 데이터 이어 붙이기(pd.concat()): 축의 로직 설정으로 이어 붙이기
print(df4), print('='*50)
print(df5), print('='*50)

result = pd.concat([df4, df5], axis=1)
print(result), print('='*50)

# reindex: DataFrame의 인덱스를 재정렬하거나 변경할 때 사용
# df1의 index를 기준으로 결합
result3 = pd.concat([df4, df5], axis=1).reindex(df4.index)
print(result3)
```

        A   B   C
    0  A0  B0  C0
    2  A1  B1  C1
    3  A2  B2  C2
    ==================================================
        B   C   E
    2  B2  C2  E2
    6  B6  C6  E6
    7  B7  C7  E7
    ==================================================
         A    B    C    B    C    E
    0   A0   B0   C0  NaN  NaN  NaN
    2   A1   B1   C1   B2   C2   E2
    3   A2   B2   C2  NaN  NaN  NaN
    6  NaN  NaN  NaN   B6   C6   E6
    7  NaN  NaN  NaN   B7   C7   E7
    ==================================================
        A   B   C    B    C    E
    0  A0  B0  C0  NaN  NaN  NaN
    2  A1  B1  C1   B2   C2   E2
    3  A2  B2  C2  NaN  NaN  NaN
    


```python
# 1. 데이터 가공

# 데이터 이어 붙이기(pd.concat()): 축의 로직 설정으로 이어 붙이기
print(df4), print('='*50)
print(df5), print('='*50)

result4 = pd.concat([df4, df5])
print(result4), print('='*50)

# ignore_index=True: 원본 인덱스를 무시하고 결합된 결과 DataFrame에
# 새로운 정수 인덱스(0부터 시작)를 자동으로 생성
# 인덱스가 중복되는 경우 무시할 수 있음
result5 = pd.concat([df4, df5], ignore_index=True)
print(result5)
```

        A   B   C
    0  A0  B0  C0
    2  A1  B1  C1
    3  A2  B2  C2
    ==================================================
        B   C   E
    2  B2  C2  E2
    6  B6  C6  E6
    7  B7  C7  E7
    ==================================================
         A   B   C    E
    0   A0  B0  C0  NaN
    2   A1  B1  C1  NaN
    3   A2  B2  C2  NaN
    2  NaN  B2  C2   E2
    6  NaN  B6  C6   E6
    7  NaN  B7  C7   E7
    ==================================================
         A   B   C    E
    0   A0  B0  C0  NaN
    1   A1  B1  C1  NaN
    2   A2  B2  C2  NaN
    3  NaN  B2  C2   E2
    4  NaN  B6  C6   E6
    5  NaN  B7  C7   E7
    


```python
print(pd.__version__)
```

    2.2.2
    


```python
# Pandas 2.0 이상 버전에서는 DataFrame.append()와 Series.append() 메소드가
# 공식적으로 제거(removed) 되었음
result = df1.append(df2)
print(result), print('='*50)
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-50-b544df983630> in <cell line: 0>()
          6                     'C': ['C3', 'C4', 'C5']}, index=[3, 4, 5])
          7 
    ----> 8 result = df1.append(df2)
          9 print(result), print('='*50)
    

    /usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __getattr__(self, name)
       6297         ):
       6298             return self[name]
    -> 6299         return object.__getattribute__(self, name)
       6300 
       6301     @final
    

    AttributeError: 'DataFrame' object has no attribute 'append'



```python
# 1. 데이터 가공

# 데이터 이어 붙이기(pd.concat()): 차원이 다른 시리즈와 데이터프레임 이어 붙이기
df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                    'B': ['B0', 'B1', 'B2'],
                    'C': ['C0', 'C1', 'C2']}, index=[0, 1, 2])

s2 = pd.Series(['Z0', 'Z1', 'Z2', 'Z3'], name='Z')

print(df1), print('='*30)
print(s2), print('='*30)

result = pd.concat([df1, s2], axis=1)
print(result), print('='*30)

s3 = pd.Series(['*0', '*1', '*2'])  # 컬럼 이름이 없음
print(s3), print('='*30)

result2 = pd.concat([df1, s3, s3, s3], axis=1)
print(result2)
```

        A   B   C
    0  A0  B0  C0
    1  A1  B1  C1
    2  A2  B2  C2
    ==============================
    0    Z0
    1    Z1
    2    Z2
    3    Z3
    Name: Z, dtype: object
    ==============================
         A    B    C   Z
    0   A0   B0   C0  Z0
    1   A1   B1   C1  Z1
    2   A2   B2   C2  Z2
    3  NaN  NaN  NaN  Z3
    ==============================
    0    *0
    1    *1
    2    *2
    dtype: object
    ==============================
        A   B   C   0   1   2
    0  A0  B0  C0  *0  *0  *0
    1  A1  B1  C1  *1  *1  *1
    2  A2  B2  C2  *2  *2  *2
    


```python
# 1. 데이터 가공

# 데이터 이어 붙이기(pd.concat()): 그룹 키로 이어 붙이기
s4 = pd.Series([0, 1, 2, 3], name='J')  # 컬럼 이름 J
s5 = pd.Series([0, 1, 2, 3])            # 컬럼 이름이 없음
s6 = pd.Series([0, 1, 4, 5])            # 컬럼 이름이 없음

result3 = pd.concat([s4, s5, s6], axis=1)
print(result3), print('='*30)

# keys = []: DataFrame의 컬럼 이름을 명시적으로 지정
result4 = pd.concat([s4, s5, s6], axis=1, keys=['ha', 'hi', 'ho'])
print(result4)
```

       J  0  1
    0  0  0  0
    1  1  1  1
    2  2  2  4
    3  3  3  5
    ==============================
       ha  hi  ho
    0   0   0   0
    1   1   1   1
    2   2   2   4
    3   3   3   5
    


```python
# 1. 데이터 가공

# 데이터 이어 붙이기(pd.concat()): 그룹 키로 이어 붙이기
df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                    'B': ['B0', 'B1', 'B2'],
                    'C': ['C0', 'C1', 'C2']}, index=[0, 1, 2])
df2 = pd.DataFrame({'A': ['A3', 'A4', 'A5'],
                    'B': ['B3', 'B4', 'B5'],
                    'C': ['C3', 'C4', 'C5']}, index=[3, 4, 5])
df3 = pd.DataFrame({'A': ['A6', 'A7', 'A8'],
                    'B': ['B6', 'B7', 'B8'],
                    'C': ['C6', 'C7', 'C8']}, index=[6, 7, 8])
frames = [df1, df2, df3]

print(df1), print('='*30)
print(df2), print('='*30)
print(df3), print('='*30)

result = pd.concat(frames, keys=['ha', 'hi', 'ho'])
print(result), print('='*50)

# DataFrame들을 딕셔너리 형태로 준비
# 딕셔너리로 objs를 전달할 경우 딕셔너리의 키가 keys 역할을 함
# 즉, 딕셔너리의 키('ha', 'hi', 'ho')가 해당 DataFrame의 상위 레벨 인덱스가 됨
pic = {'ha': df1, 'hi': df2, 'ho': df3}
result1 = pd.concat(pic)
print(result1), print('='*50)

# 딕셔너리의 키들 중 특정 키만 선택하여 포함하고, 그 순서도 재정렬
result2 = pd.concat(pic, keys=['ho', 'hi'])
print(result2)
```

        A   B   C
    0  A0  B0  C0
    1  A1  B1  C1
    2  A2  B2  C2
    ==============================
        A   B   C
    3  A3  B3  C3
    4  A4  B4  C4
    5  A5  B5  C5
    ==============================
        A   B   C
    6  A6  B6  C6
    7  A7  B7  C7
    8  A8  B8  C8
    ==============================
           A   B   C
    ha 0  A0  B0  C0
       1  A1  B1  C1
       2  A2  B2  C2
    hi 3  A3  B3  C3
       4  A4  B4  C4
       5  A5  B5  C5
    ho 6  A6  B6  C6
       7  A7  B7  C7
       8  A8  B8  C8
    ==================================================
           A   B   C
    ha 0  A0  B0  C0
       1  A1  B1  C1
       2  A2  B2  C2
    hi 3  A3  B3  C3
       4  A4  B4  C4
       5  A5  B5  C5
    ho 6  A6  B6  C6
       7  A7  B7  C7
       8  A8  B8  C8
    ==================================================
           A   B   C
    ho 6  A6  B6  C6
       7  A7  B7  C7
       8  A8  B8  C8
    hi 3  A3  B3  C3
       4  A4  B4  C4
       5  A5  B5  C5
    


```python

```


```python
# 1. 데이터 가공

# 데이터베이스 타입의 데이터프레임 또는 시리즈 합치기: merge()
df1 = pd.DataFrame({'key_name': ['A', 'B', 'C', 'D'], 'value': np.random.randn(4)})
df2 = pd.DataFrame({'key_name': ['B', 'D', 'D', 'E'], 'value': np.random.randn(4)})
print(df1), print('='*30)
print(df2), print('='*30)

# how 매개변수가 지정되지 않았으므로, pd.merge()는 기본값인 how='inner' (내부 조인) 사용
# 내부 조인은 on으로 지정된 키('key_name')가 양쪽 DataFrame에 모두 존재하는 경우에만
# 해당 행들을 결과에 포함.
# 조인 키('key_name')를 제외하고 이름이 중복되는 컬럼에 대해 자동으로 접미사(_x와 _y)를 붙여 구분
print(pd.merge(df1, df2, on='key_name'))
```

      key_name     value
    0        A -1.557818
    1        B  0.524327
    2        C  0.105372
    3        D -0.489903
    ==============================
      key_name     value
    0        B  1.525080
    1        D  0.055619
    2        D -0.979631
    3        E -0.281685
    ==============================
      key_name   value_x   value_y
    0        B  0.524327  1.525080
    1        D -0.489903  0.055619
    2        D -0.489903 -0.979631
    


```python
# 1. 데이터 가공

# 데이터베이스 타입의 데이터프레임 또는 시리즈 합치기: merge()
left = pd.merge(df1, df2, on='key_name', how='left')
print(left), print('='*30)

right = pd.merge(df1, df2, on='key_name', how='right')
print(right), print('='*30)

outer = pd.merge(df1, df2, on='key_name', how='outer')
print(outer), print('='*30)

inner = pd.merge(df1, df2, on='key_name', how='inner')  # how='inner'가 기본값
print(inner)
```

      key_name   value_x   value_y
    0        A -1.557818       NaN
    1        B  0.524327  1.525080
    2        C  0.105372       NaN
    3        D -0.489903  0.055619
    4        D -0.489903 -0.979631
    ==============================
      key_name   value_x   value_y
    0        B  0.524327  1.525080
    1        D -0.489903  0.055619
    2        D -0.489903 -0.979631
    3        E       NaN -0.281685
    ==============================
      key_name   value_x   value_y
    0        A -1.557818       NaN
    1        B  0.524327  1.525080
    2        C  0.105372       NaN
    3        D -0.489903  0.055619
    4        D -0.489903 -0.979631
    5        E       NaN -0.281685
    ==============================
      key_name   value_x   value_y
    0        B  0.524327  1.525080
    1        D -0.489903  0.055619
    2        D -0.489903 -0.979631
    


```python
# 1. 데이터 가공

# 데이터베이스 타입의 데이터프레임 또는 시리즈 합치기: merge()
left = pd.DataFrame({'key1': ['Z0', 'Z0', 'Z1', 'Z2'],
                     'key2': ['Z0', 'Z1', 'Z0', 'Z1'],
                     'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3']})
right = pd.DataFrame({'key1': ['Z0', 'Z1', 'Z1', 'Z2'],
                      'key2': ['Z0', 'Z0', 'Z0', 'Z0'],
                      'A': ['C0', 'C1', 'C2', 'C3'],
                      'D': ['D0', 'D1', 'D2', 'D3']})

print(left), print('='*30)
print(right), print('='*30)

# left DataFrame의 ('key1','key2') 조합 : ('Z0','Z0'), ('Z0','Z1'), ('Z1','Z0'), ('Z2','Z1')
# right DataFrame의 ('key1','key2') 조합: ('Z0','Z0'), ('Z1','Z0'), ('Z1','Z0'), ('Z2','Z0')
result = pd.merge(left, right, on=['key1', 'key2']) # how='inner'가 기본값
print(result)
```

      key1 key2   A   B
    0   Z0   Z0  A0  B0
    1   Z0   Z1  A1  B1
    2   Z1   Z0  A2  B2
    3   Z2   Z1  A3  B3
    ==============================
      key1 key2   A   D
    0   Z0   Z0  C0  D0
    1   Z1   Z0  C1  D1
    2   Z1   Z0  C2  D2
    3   Z2   Z0  C3  D3
    ==============================
      key1 key2 A_x   B A_y   D
    0   Z0   Z0  A0  B0  C0  D0
    1   Z1   Z0  A2  B2  C1  D1
    2   Z1   Z0  A2  B2  C2  D2
    


```python
# 1. 데이터 가공

# 데이터베이스 타입의 데이터프레임 또는 시리즈 합치기: merge()
print(left), print('='*30)
print(right), print('='*30)

# left DataFrame의 ('key1','key2') 조합 : ('Z0','Z0'), ('Z0','Z1'), ('Z1','Z0'), ('Z2','Z1')
# right DataFrame의 ('key1','key2') 조합: ('Z0','Z0'), ('Z1','Z0'), ('Z1','Z0'), ('Z2','Z0')
result = pd.merge(left, right, how='left', on=['key1', 'key2'])
print(result), print('='*30)

result = pd.merge(left, right, how='right', on=['key1', 'key2'])
print(result), print('='*30)

result = pd.merge(left, right, how='outer', on=['key1', 'key2'])
print(result)
```

      key1 key2   A   B
    0   Z0   Z0  A0  B0
    1   Z0   Z1  A1  B1
    2   Z1   Z0  A2  B2
    3   Z2   Z1  A3  B3
    ==============================
      key1 key2   A   D
    0   Z0   Z0  C0  D0
    1   Z1   Z0  C1  D1
    2   Z1   Z0  C2  D2
    3   Z2   Z0  C3  D3
    ==============================
      key1 key2 A_x   B  A_y    D
    0   Z0   Z0  A0  B0   C0   D0
    1   Z0   Z1  A1  B1  NaN  NaN
    2   Z1   Z0  A2  B2   C1   D1
    3   Z1   Z0  A2  B2   C2   D2
    4   Z2   Z1  A3  B3  NaN  NaN
    ==============================
      key1 key2  A_x    B A_y   D
    0   Z0   Z0   A0   B0  C0  D0
    1   Z1   Z0   A2   B2  C1  D1
    2   Z1   Z0   A2   B2  C2  D2
    3   Z2   Z0  NaN  NaN  C3  D3
    ==============================
      key1 key2  A_x    B  A_y    D
    0   Z0   Z0   A0   B0   C0   D0
    1   Z0   Z1   A1   B1  NaN  NaN
    2   Z1   Z0   A2   B2   C1   D1
    3   Z1   Z0   A2   B2   C2   D2
    4   Z2   Z0  NaN  NaN   C3   D3
    5   Z2   Z1   A3   B3  NaN  NaN
    


```python
# 1. 데이터 가공

# 데이터베이스 타입의 데이터프레임 또는 시리즈 합치기: join() 메소드
left = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                     'B': ['B0', 'B1', 'B2']}, index=['Z0', 'Z1', 'Z2'])
right = pd.DataFrame({'C': ['C0', 'C2', 'C3'],
                      'D': ['D0', 'D2', 'D3']}, index=['Z0', 'Z2', 'Z3'])

print(left), print('='*30)
print(right), print('='*30)

result = left.join(right)
#result = left.join(right, how='left')
print(result), print('='*30)

result = left.join(right, how='outer')
print(result)
```

         A   B
    Z0  A0  B0
    Z1  A1  B1
    Z2  A2  B2
    ==============================
         C   D
    Z0  C0  D0
    Z2  C2  D2
    Z3  C3  D3
    ==============================
         A   B    C    D
    Z0  A0  B0   C0   D0
    Z1  A1  B1  NaN  NaN
    Z2  A2  B2   C2   D2
    ==============================
          A    B    C    D
    Z0   A0   B0   C0   D0
    Z1   A1   B1  NaN  NaN
    Z2   A2   B2   C2   D2
    Z3  NaN  NaN   C3   D3
    


```python
# 1. 데이터 가공

# 데이터베이스 타입의 데이터프레임 또는 시리즈 합치기: join() 메소드
print(left), print('='*30)
print(right), print('='*30)

result = left.join(right, how='right')
print(result), print('='*30)

result = left.join(right, how='inner')
print(result)
```

         A   B
    Z0  A0  B0
    Z1  A1  B1
    Z2  A2  B2
    ==============================
         C   D
    Z0  C0  D0
    Z2  C2  D2
    Z3  C3  D3
    ==============================
          A    B   C   D
    Z0   A0   B0  C0  D0
    Z2   A2   B2  C2  D2
    Z3  NaN  NaN  C3  D3
    ==============================
         A   B   C   D
    Z0  A0  B0  C0  D0
    Z2  A2  B2  C2  D2
    


```python
# 1. 데이터 가공

# 데이터베이스 타입의 데이터프레임 또는 시리즈 합치기: join() 메소드
left = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3'],
                     'key': ['Z0', 'Z1', 'Z0', 'Z1']})
right = pd.DataFrame({'C': ['C0', 'C1'],
                      'D': ['D0', 'D1']}, index=['Z0', 'Z1'])
#                       'D': ['D0', 'D1']}, index=['Z3', 'Z4'])
print(left), print('='*30)
print(right), print('='*30)

# join() 메소드는 how='left' (왼쪽 조인)를 사용
# 즉,left DataFrame의 모든 행을 유지하고, right DataFrame에서 매칭되는 행을 추가
result = left.join(right, on='key')
#result = left.join(right, on='key', how='left')
print(result)
```

        A   B key
    0  A0  B0  Z0
    1  A1  B1  Z1
    2  A2  B2  Z0
    3  A3  B3  Z1
    ==============================
         C   D
    Z0  C0  D0
    Z1  C1  D1
    ==============================
        A   B key   C   D
    0  A0  B0  Z0  C0  D0
    1  A1  B1  Z1  C1  D1
    2  A2  B2  Z0  C0  D0
    3  A3  B3  Z1  C1  D1
    


```python
# 1. 데이터 가공

# 데이터베이스 타입의 데이터프레임 또는 시리즈 합치기: 멀티인덱스 객체 합치기
left = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3'],
                     'key1': ['Z0', 'Z0', 'Z1', 'Z2'],
                     'key2': ['Z0', 'Z1', 'Z0', 'Z1']})

ind = pd.MultiIndex.from_tuples([('Z0', 'Z0'), ('Z1', 'Z0'), ('Z2', 'Z0'), ('Z2', 'Z1')])
right = pd.DataFrame({'C': ['C0', 'C1', 'C2', 'C3'],
                      'D': ['D0', 'D1', 'D2', 'D3']}, index=ind)

print(left), print('='*30)
print(right), print('='*30)

# left DataFrame의 ('key1','key2') 조합 : ('Z0','Z0'), ('Z0','Z1'), ('Z1','Z0'), ('Z2','Z1')
result = left.join(right, on=['key1', 'key2'])
print(result)
```

        A   B key1 key2
    0  A0  B0   Z0   Z0
    1  A1  B1   Z0   Z1
    2  A2  B2   Z1   Z0
    3  A3  B3   Z2   Z1
    ==============================
            C   D
    Z0 Z0  C0  D0
    Z1 Z0  C1  D1
    Z2 Z0  C2  D2
       Z1  C3  D3
    ==============================
        A   B key1 key2    C    D
    0  A0  B0   Z0   Z0   C0   D0
    1  A1  B1   Z0   Z1  NaN  NaN
    2  A2  B2   Z1   Z0   C1   D1
    3  A3  B3   Z2   Z1   C3   D3
    


```python
# 1. 데이터 가공

# 데이터베이스 타입의 데이터프레임 또는 시리즈 합치기: 멀티인덱스 객체 합치기
left = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                     'B': ['B0', 'B1', 'B2']},
                     index=pd.Index(['Z0', 'Z1', 'Z2'], name='key'))

ind = pd.MultiIndex.from_tuples([('Z0','Y0'),('Z1','Y1'),('Z2','Y2'),('Z2','Y3')],
                                names=['key', 'Y'])
right = pd.DataFrame({'C':['C0', 'C1', 'C2', 'C3'],
                      'D':['D0', 'D1', 'D2', 'D3']},index=ind)

print(left), print('='*30)
print(right), print('='*30)

# 멀티인덱스를 가지는 데이터프레임을 하나의 인덱스를 가지는 데이터프레임과 합칠 수 있음
# 이때 레벨은 하나의 인덱스를 가지는 데이터프레임의 인덱스 이름을 기준으로 결정
result = left.join(right, how='inner')
#result = left.join(right, how='left') # how='left' 기본값
print(result)
```

          A   B
    key        
    Z0   A0  B0
    Z1   A1  B1
    Z2   A2  B2
    ==============================
             C   D
    key Y         
    Z0  Y0  C0  D0
    Z1  Y1  C1  D1
    Z2  Y2  C2  D2
        Y3  C3  D3
    ==============================
             A   B   C   D
    key Y                 
    Z0  Y0  A0  B0  C0  D0
    Z1  Y1  A1  B1  C1  D1
    Z2  Y2  A2  B2  C2  D2
        Y3  A2  B2  C3  D3
    


```python
# 1. 데이터 가공

# 데이터베이스 타입의 데이터프레임 또는 시리즈 합치기: 멀티인덱스 객체 합치기

# 위 셀과 같은 결과임
result = pd.merge(left.reset_index(), right.reset_index(),
                  on=['key'], how='inner').set_index(['key', 'Y'])

print(left.reset_index()), print('='*30)
print(right.reset_index()), print('='*30)
print(result)
```

      key   A   B
    0  Z0  A0  B0
    1  Z1  A1  B1
    2  Z2  A2  B2
    ==============================
      key   Y   C   D
    0  Z0  Y0  C0  D0
    1  Z1  Y1  C1  D1
    2  Z2  Y2  C2  D2
    3  Z2  Y3  C3  D3
    ==============================
             A   B   C   D
    key Y                 
    Z0  Y0  A0  B0  C0  D0
    Z1  Y1  A1  B1  C1  D1
    Z2  Y2  A2  B2  C2  D2
        Y3  A2  B2  C3  D3
    


```python
# 1. 데이터 가공

# 데이터베이스 타입의 데이터프레임 또는 시리즈 합치기: 멀티인덱스 객체 합치기
l_ind = pd.MultiIndex.from_product([list('abc'), list('xy'), [1, 2]],
                                   names=['abc', 'xy', 'num'])
left = pd.DataFrame({'z1': range(12)}, index=l_ind)

r_ind = pd.MultiIndex.from_product([list('abc'), list('xy')], names=['abc','xy'])
right = pd.DataFrame({'z2': [100 * i for i in range(1, 7)]}, index=r_ind)

print(left), print('='*30)
print(right), print('='*30)

result = left.join(right, on=['abc', 'xy'], how='inner')
print(result)
```

                z1
    abc xy num    
    a   x  1     0
           2     1
        y  1     2
           2     3
    b   x  1     4
           2     5
        y  1     6
           2     7
    c   x  1     8
           2     9
        y  1    10
           2    11
    ==============================
             z2
    abc xy     
    a   x   100
        y   200
    b   x   300
        y   400
    c   x   500
        y   600
    ==============================
                z1   z2
    abc xy num         
    a   x  1     0  100
           2     1  100
        y  1     2  200
           2     3  200
    b   x  1     4  300
           2     5  300
        y  1     6  400
           2     7  400
    c   x  1     8  500
           2     9  500
        y  1    10  600
           2    11  600
    


```python
# 1. 데이터 가공

# 데이터베이스 타입의 데이터프레임 또는 시리즈 합치기: 열과 인덱스 레벨을 조합해 합치기
l_ind = pd.Index(['Z0', 'Z0', 'Z1', 'Z2'], name='key1')
left = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3'],
                     'key2': ['Z0', 'Z1', 'Z0', 'Z1']}, index=l_ind)

r_ind = pd.Index(['Z0', 'Z1', 'Z2', 'Z2'], name='key1')
right = pd.DataFrame({'C': ['C0', 'C1', 'C2', 'C3'],
                      'D': ['D0', 'D1', 'D2', 'D3'],
                      'key2': ['Z0', 'Z0', 'Z0', 'Z1']}, index=r_ind)

print(left), print('='*30)
print(right), print('='*30)

# left DataFrame의 ('key1','key2') 조합 : ('Z0','Z0'), ('Z0','Z1'), ('Z1','Z0'), ('Z2','Z1')
# right DataFrame의 ('key1','key2') 조합: ('Z0','Z0'), ('Z1','Z0'), ('Z2','Z0'), ('Z2','Z1')
# how='inner' (기본값)이므로 on으로 지정된 모든 조인 키(key1, key2의 조합)가
# 양쪽 DataFrame에 모두 존재하는 경우에만 해당 행들을 결과에 포함
result = left.merge(right, on=['key1', 'key2'])
print(result), print('='*30)

result1 = pd.merge(left, right, on=['key1', 'key2']) # 위와 같은 결과임
print(result1)
```

           A   B key2
    key1             
    Z0    A0  B0   Z0
    Z0    A1  B1   Z1
    Z1    A2  B2   Z0
    Z2    A3  B3   Z1
    ==============================
           C   D key2
    key1             
    Z0    C0  D0   Z0
    Z1    C1  D1   Z0
    Z2    C2  D2   Z0
    Z2    C3  D3   Z1
    ==============================
           A   B key2   C   D
    key1                     
    Z0    A0  B0   Z0  C0  D0
    Z1    A2  B2   Z0  C1  D1
    Z2    A3  B3   Z1  C3  D3
    ==============================
           A   B key2   C   D
    key1                     
    Z0    A0  B0   Z0  C0  D0
    Z1    A2  B2   Z0  C1  D1
    Z2    A3  B3   Z1  C3  D3
    


```python
# 1. 데이터 가공

# 데이터베이스 타입의 데이터프레임 또는 시리즈 합치기: 중복되는 열 처리하기
left = pd.DataFrame({'z': ['Z0', 'Z1', 'Z2'], 'v': [1, 2, 3]})
right = pd.DataFrame({'z': ['Z0', 'Z0', 'Z3'], 'v': [4, 5, 6]})

print(left), print('='*30)
print(right), print('='*30)

result = pd.merge(left, right, on='z') # how='inner'
result1 = pd.merge(left, right, on='z', suffixes=['_l', '_r']) # how='inner'
print(result), print('='*30)
print(result1)
```

        z  v
    0  Z0  1
    1  Z1  2
    2  Z2  3
    ==============================
        z  v
    0  Z0  4
    1  Z0  5
    2  Z3  6
    ==============================
        z  v_x  v_y
    0  Z0    1    4
    1  Z0    1    5
    ==============================
        z  v_l  v_r
    0  Z0    1    4
    1  Z0    1    5
    


```python
# 1. 데이터 가공

# 데이터베이스 타입의 데이터프레임 또는 시리즈 합치기: 중복되는 열 처리하기
print(left), print('='*30)
print(right), print('='*30)

left = left.set_index('v')
right = right.set_index('v')

print(left), print('='*30)
print(right), print('='*30)

result = left.join(right, lsuffix='_l', rsuffix='_r')
print(result)
```

        z  v
    0  Z0  1
    1  Z1  2
    2  Z2  3
    ==============================
        z  v
    0  Z0  4
    1  Z0  5
    2  Z3  6
    ==============================
        z
    v    
    1  Z0
    2  Z1
    3  Z2
    ==============================
        z
    v    
    4  Z0
    5  Z0
    6  Z3
    ==============================
      z_l  z_r
    v         
    1  Z0  NaN
    2  Z1  NaN
    3  Z2  NaN
    


```python
# 1. 데이터 가공

# 데이터베이스 타입의 데이터프레임 또는 시리즈 합치기: 시리즈나 데이터프레임의 열 안에서 값을 합치기
df1 = pd.DataFrame([[np.nan, 3., 5.], [-4.6, np.nan, np.nan], [np.nan, 7., np.nan]])
df2 = pd.DataFrame([[-2.6, np.nan, -8.2], [-5., 1.6, 4]], index=[1, 2])

print(df1), print('='*30)
print(df2), print('='*30)

result1 = df1.combine_first(df2)
result2 = df2.combine_first(df1)
print(result1), print('='*30)
print(result2), print('='*30)

df1.update(df2) # df2.combine_first(df1)과 결과가 같음
print(df1)
```

         0    1    2
    0  NaN  3.0  5.0
    1 -4.6  NaN  NaN
    2  NaN  7.0  NaN
    ==============================
         0    1    2
    1 -2.6  NaN -8.2
    2 -5.0  1.6  4.0
    ==============================
         0    1    2
    0  NaN  3.0  5.0
    1 -4.6  NaN -8.2
    2 -5.0  7.0  4.0
    ==============================
         0    1    2
    0  NaN  3.0  5.0
    1 -2.6  NaN -8.2
    2 -5.0  1.6  4.0
    ==============================
         0    1    2
    0  NaN  3.0  5.0
    1 -2.6  NaN -8.2
    2 -5.0  1.6  4.0
    


```python
# 데이터 가공

# 데이터 재형성하기: 데이터프레임 객체 피벗 pivot()
data = {'name': ['haena', 'naeun', 'una', 'bum', 'suho'],
        'type': ['tennis', 'tennis', 'swim', 'swim', 'tennis'],
        'records': ['A', 'B', 'C', 'A', 'B'],
        'sex': ['F', 'F', 'F', 'M', 'M'],
        'period': [3, 3, 1, 5, 2]}

df = pd.DataFrame(data)
print(df), print('='*50)

dfp = df.pivot(index='name', columns='type', values=['records', 'sex'])
print(dfp), print('='*50)

dfp2 = df.pivot(index='name', columns='type')
print(dfp2)
```

        name    type records sex  period
    0  haena  tennis       A   F       3
    1  naeun  tennis       B   F       3
    2    una    swim       C   F       1
    3    bum    swim       A   M       5
    4   suho  tennis       B   M       2
    ==================================================
          records         sex       
    type     swim tennis swim tennis
    name                            
    bum         A    NaN    M    NaN
    haena     NaN      A  NaN      F
    naeun     NaN      B  NaN      F
    suho      NaN      B  NaN      M
    una         C    NaN    F    NaN
    ==================================================
          records         sex        period       
    type     swim tennis swim tennis   swim tennis
    name                                          
    bum         A    NaN    M    NaN    5.0    NaN
    haena     NaN      A  NaN      F    NaN    3.0
    naeun     NaN      B  NaN      F    NaN    3.0
    suho      NaN      B  NaN      M    NaN    2.0
    una         C    NaN    F    NaN    1.0    NaN
    


```python
# 데이터 가공

# 데이터 재형성하기: 데이터프레임 객체 피벗 pivot_table()
print(df), print('='*50)

dfp = df.pivot_table(index='type', columns='records', values='period', aggfunc=np.max)
print(dfp)
```

        name    type records sex  period
    0  haena  tennis       A   F       3
    1  naeun  tennis       B   F       3
    2    una    swim       C   F       1
    3    bum    swim       A   M       5
    4   suho  tennis       B   M       2
    ==================================================
    records    A    B    C
    type                  
    swim     5.0  NaN  1.0
    tennis   3.0  3.0  NaN
    

    <ipython-input-40-ae73fac17e49>:6: FutureWarning: The provided callable <function max at 0x7ebdacfb93a0> is currently using DataFrameGroupBy.max. In a future version of pandas, the provided callable will be used directly. To keep current behavior pass the string "max" instead.
      dfp = df.pivot_table(index='type', columns='records', values='period', aggfunc=np.max)
    


```python
# 데이터 가공

# 데이터 재형성하기: 데이터프레임 객체 피벗 pivot_table()
import datetime
df = pd.DataFrame({'A': ['one', 'one', 'two', 'three'] * 2,
                   'B': ['x', 'y'] * 4,
                   'C': ['ha', 'ha', 'ha', 'hi', 'hi', 'hi', 'ha', 'hi'],
                   'D': np.arange(8),
                   'E': [datetime.datetime(2021, i, 1) for i in range(1, 9)]})

print(df), print('='*50)

dfp = df.pivot_table(values='D', index=['A', 'B'], columns=['C']) # aggfunc='mean' (기본값)
print(dfp), print('='*50)

dfp2 = pd.pivot_table(df, values='D', index=['B'], columns=['A','C'], aggfunc=np.sum)
print(dfp2), print('='*50)

str_df = dfp2.to_string(na_rep='')
print(str_df)
```

           A  B   C  D          E
    0    one  x  ha  0 2021-01-01
    1    one  y  ha  1 2021-02-01
    2    two  x  ha  2 2021-03-01
    3  three  y  hi  3 2021-04-01
    4    one  x  hi  4 2021-05-01
    5    one  y  hi  5 2021-06-01
    6    two  x  ha  6 2021-07-01
    7  three  y  hi  7 2021-08-01
    ==================================================
    C         ha   hi
    A     B          
    one   x  0.0  4.0
          y  1.0  5.0
    three y  NaN  5.0
    two   x  4.0  NaN
    ==================================================
    A  one      three  two
    C   ha   hi    hi   ha
    B                     
    x  0.0  4.0   NaN  8.0
    y  1.0  5.0  10.0  NaN
    ==================================================
    A  one      three  two
    C   ha   hi    hi   ha
    B                     
    x  0.0  4.0        8.0
    y  1.0  5.0  10.0     
    

    <ipython-input-41-3a06daabe38f>:16: FutureWarning: The provided callable <function sum at 0x7ebdacfb8cc0> is currently using DataFrameGroupBy.sum. In a future version of pandas, the provided callable will be used directly. To keep current behavior pass the string "sum" instead.
      dfp2 = pd.pivot_table(df, values='D', index=['B'], columns=['A','C'], aggfunc=np.sum)
    


```python
# 데이터 가공

# 데이터 재형성하기: 교차표 crosstab()
ha, hi, top, down, one, two = 'ha', 'hi', 'top', 'down', 'one', 'two'
a = np.array([ha,  ha,   hi,  hi,  ha,  ha], dtype=object)
b = np.array([one, one, two,  one, two, one], dtype=object)
c = np.array([top, top, down, top, top, down], dtype=object)

dfc = pd.crosstab(a, [b, c], rownames=['a'], colnames=['b', 'c'])
print(dfc)
```

    b   one      two    
    c  down top down top
    a                   
    ha    1   2    0   1
    hi    0   1    1   0
    


```python
# 데이터 가공

# 데이터 재형성하기: 교차표 crosstab()
df = pd.DataFrame({'A': [1, 2, 2, 2, 2], 'B': [3, 3, 7, 7, 7], 'C': [1, 1, np.nan, 1, 1]})
print(df), print('='*20)

dfc = pd.crosstab(df.A, df.B)
print(dfc), print('='*20)

dfc2 = pd.crosstab(df.A, df.B, normalize=True)
print(dfc2), print('='*20)

dfc2 = pd.crosstab(df.A, df.B, normalize='columns')
print(dfc2), print('='*20)

dfc2 = pd.crosstab(df.A, df.B, normalize='index')
print(dfc2)
```

       A  B    C
    0  1  3  1.0
    1  2  3  1.0
    2  2  7  NaN
    3  2  7  1.0
    4  2  7  1.0
    ====================
    B  3  7
    A      
    1  1  0
    2  1  3
    ====================
    B    3    7
    A          
    1  0.2  0.0
    2  0.2  0.6
    ====================
    B    3    7
    A          
    1  0.5  0.0
    2  0.5  1.0
    ====================
    B     3     7
    A            
    1  1.00  0.00
    2  0.25  0.75
    


```python
# 데이터 가공

# 데이터 재형성하기: 교차표 crosstab()
df = pd.DataFrame({'A': [1, 2, 2, 2, 2], 'B': [3, 3, 7, 7, 7], 'C': [1, 1, np.nan, 1, 1]})
print(df), print('='*20)

dfc = pd.crosstab(df.A, df.B, values=df.C, aggfunc=np.sum)
print(dfc), print('='*20)

dfc2 = pd.crosstab(df.A, df.B, values=df.C, aggfunc=np.sum, margins=True)
print(dfc2), print('='*20)

dfc3 = pd.crosstab(df.A, df.B, values=df.C, aggfunc=np.sum, normalize=True, margins=True)
print(dfc3)
```

       A  B    C
    0  1  3  1.0
    1  2  3  1.0
    2  2  7  NaN
    3  2  7  1.0
    4  2  7  1.0
    ====================
    B    3    7
    A          
    1  1.0  NaN
    2  1.0  2.0
    ====================
    B      3    7  All
    A                 
    1    1.0  NaN  1.0
    2    1.0  2.0  3.0
    All  2.0  2.0  4.0
    ====================
    B       3    7   All
    A                   
    1    0.25  0.0  0.25
    2    0.25  0.5  0.75
    All  0.50  0.5  1.00
    

    <ipython-input-64-0db1b58b1d10>:7: FutureWarning: The provided callable <function sum at 0x7bd258ff8cc0> is currently using DataFrameGroupBy.sum. In a future version of pandas, the provided callable will be used directly. To keep current behavior pass the string "sum" instead.
      dfc = pd.crosstab(df.A, df.B, values=df.C, aggfunc=np.sum)
    <ipython-input-64-0db1b58b1d10>:10: FutureWarning: The provided callable <function sum at 0x7bd258ff8cc0> is currently using DataFrameGroupBy.sum. In a future version of pandas, the provided callable will be used directly. To keep current behavior pass the string "sum" instead.
      dfc2 = pd.crosstab(df.A, df.B, values=df.C, aggfunc=np.sum, margins=True)
    <ipython-input-64-0db1b58b1d10>:10: FutureWarning: The provided callable <function sum at 0x7bd258ff8cc0> is currently using DataFrameGroupBy.sum. In a future version of pandas, the provided callable will be used directly. To keep current behavior pass the string "sum" instead.
      dfc2 = pd.crosstab(df.A, df.B, values=df.C, aggfunc=np.sum, margins=True)
    <ipython-input-64-0db1b58b1d10>:10: FutureWarning: The provided callable <function sum at 0x7bd258ff8cc0> is currently using DataFrameGroupBy.sum. In a future version of pandas, the provided callable will be used directly. To keep current behavior pass the string "sum" instead.
      dfc2 = pd.crosstab(df.A, df.B, values=df.C, aggfunc=np.sum, margins=True)
    <ipython-input-64-0db1b58b1d10>:13: FutureWarning: The provided callable <function sum at 0x7bd258ff8cc0> is currently using DataFrameGroupBy.sum. In a future version of pandas, the provided callable will be used directly. To keep current behavior pass the string "sum" instead.
      dfc3 = pd.crosstab(df.A, df.B, values=df.C, aggfunc=np.sum, normalize=True, margins=True)
    <ipython-input-64-0db1b58b1d10>:13: FutureWarning: The provided callable <function sum at 0x7bd258ff8cc0> is currently using DataFrameGroupBy.sum. In a future version of pandas, the provided callable will be used directly. To keep current behavior pass the string "sum" instead.
      dfc3 = pd.crosstab(df.A, df.B, values=df.C, aggfunc=np.sum, normalize=True, margins=True)
    <ipython-input-64-0db1b58b1d10>:13: FutureWarning: The provided callable <function sum at 0x7bd258ff8cc0> is currently using DataFrameGroupBy.sum. In a future version of pandas, the provided callable will be used directly. To keep current behavior pass the string "sum" instead.
      dfc3 = pd.crosstab(df.A, df.B, values=df.C, aggfunc=np.sum, normalize=True, margins=True)
    


```python

```


```python

```


```python
import numpy as np
import pandas as pd
```


```python
# 2. 데이터의 그룹 연산

# 데이터 객체를 그룹 연산: GroupBy 객체 속성
df = pd.DataFrame({'A': ['ha', 'hi', 'ho', 'ha', 'ho'],
                   'B': ['one', 'two', 'one', 'one', 'two'],
                   'Data1': np.random.randn(5),
                   'Data2': np.random.randn(5)})
print(df), print('='*50)

grouped1 = df.groupby('A')

# grouped1은 DataFrameGroupBy 객체임
# 그룹들의 속성은 키가 반영된 단일 그룹임
print(grouped1)
gr_dict = dict(list(grouped1))
print(dict(list(grouped1))), print('='*50)

print(grouped1.groups)
# print(grouped1.mean())
```

        A    B     Data1     Data2
    0  ha  one -0.790777 -1.323456
    1  hi  two  0.119117 -1.630244
    2  ho  one  0.692038 -0.464918
    3  ha  one  0.350813 -0.763992
    4  ho  two -1.439306 -0.575059
    ==================================================
    <pandas.core.groupby.generic.DataFrameGroupBy object at 0x7851720eb910>
    {'ha':     A    B     Data1     Data2
    0  ha  one -0.790777 -1.323456
    3  ha  one  0.350813 -0.763992, 'hi':     A    B     Data1     Data2
    1  hi  two  0.119117 -1.630244, 'ho':     A    B     Data1     Data2
    2  ho  one  0.692038 -0.464918
    4  ho  two -1.439306 -0.575059}
    ==================================================
    {'ha': [0, 3], 'hi': [1], 'ho': [2, 4]}
    


```python
# 2. 데이터의 그룹 연산

# 데이터 객체를 그룹 연산: GroupBy 객체 속성

# 특정 그룹 (예: 'ho')은 선택해 해당 데이터를 구하는 방법1
print(gr_dict['ho']), print('='*50)

# 특정 그룹 (예: 'ho')은 선택해 해당 데이터를 구하는 방법2
print(grouped1.get_group('ho'))

```

        A    B     Data1     Data2
    2  ho  one  1.217197  1.095655
    4  ho  two  0.273442 -0.497081
    ==================================================
        A    B     Data1     Data2
    2  ho  one  1.217197  1.095655
    4  ho  two  0.273442 -0.497081
    


```python
# 2. 데이터의 그룹 연산

# 데이터 객체를 그룹 연산: GroupBy 객체 속성
print(df), print('='*50)

grouped2 = df['Data2'].groupby(df['A'])
print(grouped2.mean()), print('='*50)

grouped3 = df['Data1'].groupby([df['A'], df['B']])
print(grouped3.groups), print('='*50)
print(grouped3.mean())
```

        A    B     Data1     Data2
    0  ha  one -0.790777 -1.323456
    1  hi  two  0.119117 -1.630244
    2  ho  one  0.692038 -0.464918
    3  ha  one  0.350813 -0.763992
    4  ho  two -1.439306 -0.575059
    ==================================================
    A
    ha   -1.043724
    hi   -1.630244
    ho   -0.519988
    Name: Data2, dtype: float64
    ==================================================
    {('ha', 'one'): [0, 3], ('hi', 'two'): [1], ('ho', 'one'): [2], ('ho', 'two'): [4]}
    ==================================================
    A   B  
    ha  one   -0.219982
    hi  two    0.119117
    ho  one    0.692038
        two   -1.439306
    Name: Data1, dtype: float64
    
