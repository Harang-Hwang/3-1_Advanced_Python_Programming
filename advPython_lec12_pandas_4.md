### 12주차 강의


```python
import numpy as np
import pandas as pd
```


```python
# 3. 데이터 처리

# 손실 데이터 처리
d = {'one': [1.5, 2.2, -3.0], 'two': [1.0, -1.2, 5.0], 'three': [-1.1, 2.0, 4.0]}
df = pd.DataFrame(d, index = ['a', 'c', 'f'])
df['four'] = 'ha'
df['five'] = df['one'] > 0
print(df), print('='*30)

df1 = df.reindex(['a', 'b', 'c', 'd', 'e', 'f'])
print(df1), print('='*30)

print(df1['one']), print('='*30)

# 결측치 여부를 Boolean 마스크로 반환
print(pd.isna(df1['one'])), print('='*30)

print(df1['four'].notna()), print('='*30)

print(None == None) # None은 파이썬 내장 상수로 하나만 존재함

print(np.nan == np.nan) # NaN는 그 어떤 값과도 같지 않도록 정의됨
```

       one  two  three four   five
    a  1.5  1.0   -1.1   ha   True
    c  2.2 -1.2    2.0   ha   True
    f -3.0  5.0    4.0   ha  False
    ==============================
       one  two  three four   five
    a  1.5  1.0   -1.1   ha   True
    b  NaN  NaN    NaN  NaN    NaN
    c  2.2 -1.2    2.0   ha   True
    d  NaN  NaN    NaN  NaN    NaN
    e  NaN  NaN    NaN  NaN    NaN
    f -3.0  5.0    4.0   ha  False
    ==============================
    a    1.5
    b    NaN
    c    2.2
    d    NaN
    e    NaN
    f   -3.0
    Name: one, dtype: float64
    ==============================
    a    False
    b     True
    c    False
    d     True
    e     True
    f    False
    Name: one, dtype: bool
    ==============================
    a     True
    b    False
    c     True
    d    False
    e    False
    f     True
    Name: four, dtype: bool
    ==============================
    True
    False
    


```python
# 3. 데이터 처리

# 손실 데이터 처리: reindexed (1)
df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]}, index=['row1', 'row2'])
print("원본 DataFrame: \n", df), print('='*50)

# 새로운 인덱스 및 컬럼으로 재인덱싱
new_index = ['row1', 'row3', 'row2', 'row4']
new_columns = ['col2', 'col3', 'col1']
df2 = df.reindex(index=new_index, columns=new_columns)
print("\n새로운 인덱스 및 컬럼으로 재인덱싱: \n", df2), print('='*50)

# fill_value로 누락된 값 채우기
df2_filled = df2.reindex(index=new_index, columns=new_columns, fill_value=1e6)
print("\nfill_value로 채우기: \n", df2_filled)
```

    원본 DataFrame: 
           col1  col2
    row1     1     3
    row2     2     4
    ==================================================
    
    새로운 인덱스 및 컬럼으로 재인덱싱: 
           col2  col3  col1
    row1   3.0   NaN   1.0
    row3   NaN   NaN   NaN
    row2   4.0   NaN   2.0
    row4   NaN   NaN   NaN
    ==================================================
    
    fill_value로 채우기: 
           col2  col3  col1
    row1   3.0   NaN   1.0
    row3   NaN   NaN   NaN
    row2   4.0   NaN   2.0
    row4   NaN   NaN   NaN
    


```python
# 3. 데이터 처리

# 손실 데이터 처리: reindexed (2)
# 행 방향으로 method='bfill' (backward fill, 다음 유효값) 적용
# 참고: method='backfill'과 같음
df3 = pd.DataFrame({'price': [10, 11, np.nan, 12]}, index=[1, 2, 3, 4])
print("\ndf3: \n", df3), print('='*50)

df3_bfill = df3.reindex(range(6), method='bfill')
print("\n행 방향으로 method='bfill' 적용: \n", df3_bfill), print('='*50)

# 행 방향으로 method='ffill' (forward fill, 이전 유효값) 적용
# 참고: method='pad'와 같음
df3_ffill = df3.reindex(range(6), method='ffill')
print("\n행 방향으로 method='ffill' 적용: \n", df3_ffill)
```

    
    df3: 
        price
    1   10.0
    2   11.0
    3    NaN
    4   12.0
    ==================================================
    
    행 방향으로 method='bfill' 적용: 
        price
    0   10.0
    1   10.0
    2   11.0
    3    NaN
    4   12.0
    5    NaN
    ==================================================
    
    행 방향으로 method='ffill' 적용: 
        price
    0    NaN
    1   10.0
    2   11.0
    3    NaN
    4   12.0
    5   12.0
    


```python
# 3. 데이터 처리

# 손실 데이터 처리: 손실 데이터 계산
df1 = pd.DataFrame({'one':[1.0, 2.0, 3.0], 'two':[4.0, 5.0, 6.0]}, index=['a','b','c'])
df2 = df1.copy()
df2.loc['d'] = np.nan
df2['three'] = 2.0
df2.iloc[1:2, 1:2] = np.nan

print(df1), print('='*50)
print(df2), print('='*50)
# DataFrame간 산술 연산은 인덱스와 컬럼 라벨을 정렬 후, 대응하는 라벨에 대해 연산
print(f'df1 + df2: \n{df1 + df2}'), print('='*50)

# groupby에서 NaN는 자동으로 제외됨
# 'two' 컬럼의 값을 기준으로 df2를 그룹화
print(df2.groupby('two').mean()), print('='*50)

# 값이 비어있거나 모두 NaN인 시리즈의 합은 0이고 곱은 1임
print(pd.Series([np.nan]).sum()), print('='*50)
print(pd.Series([], dtype=object).sum()), print('='*50)

print(pd.Series([np.nan]).prod()), print('='*50)
print(pd.Series([], dtype=object).prod())
```

       one  two
    a  1.0  4.0
    b  2.0  5.0
    c  3.0  6.0
    ==================================================
       one  two  three
    a  1.0  4.0    2.0
    b  2.0  NaN    2.0
    c  3.0  6.0    2.0
    d  NaN  NaN    2.0
    ==================================================
    df1 + df2: 
       one  three   two
    a  2.0    NaN   8.0
    b  4.0    NaN   NaN
    c  6.0    NaN  12.0
    d  NaN    NaN   NaN
    ==================================================
         one  three
    two            
    4.0  1.0    2.0
    6.0  3.0    2.0
    ==================================================
    0.0
    ==================================================
    0
    ==================================================
    1.0
    ==================================================
    1
    


```python
# 3. 데이터 처리

# 손실 데이터 처리: 손실 데이터 채우기 (1)
df2 = pd.DataFrame({'one':[1.0, 2.0, 3.0], 'two':[4.0, 5.0, 6.0]}, index=['a','b','c'])
df2.loc['d'] = np.nan
df2['three'] = 2
df2.iloc[1:2, 1:2] = np.nan
print(df2), print('='*50)

print(df2.fillna(0)), print('='*50)

print(df2['one'].fillna('missing')), print('='*50)

print(df2.mean()), print('='*50)

print(df2.fillna(df2.mean())), print('='*50)

# 'pad'는 'ffill'과 같이 바로 이전의 유효한 값으로 채움
print(df2.fillna(method='pad')), print('='*50)  # FutureWarning

df2.loc['c', 'three'] = np.nan
print(df2)


```

       one  two  three
    a  1.0  4.0      2
    b  2.0  NaN      2
    c  3.0  6.0      2
    d  NaN  NaN      2
    ==================================================
       one  two  three
    a  1.0  4.0      2
    b  2.0  0.0      2
    c  3.0  6.0      2
    d  0.0  0.0      2
    ==================================================
    a        1.0
    b        2.0
    c        3.0
    d    missing
    Name: one, dtype: object
    ==================================================
    one      2.0
    two      5.0
    three    2.0
    dtype: float64
    ==================================================
       one  two  three
    a  1.0  4.0      2
    b  2.0  5.0      2
    c  3.0  6.0      2
    d  2.0  5.0      2
    ==================================================
       one  two  three
    a  1.0  4.0      2
    b  2.0  4.0      2
    c  3.0  6.0      2
    d  3.0  6.0      2
    ==================================================
       one  two  three
    a  1.0  4.0    2.0
    b  2.0  NaN    2.0
    c  3.0  6.0    NaN
    d  NaN  NaN    2.0
    

    <ipython-input-6-3c9349d61f0e>:19: FutureWarning: DataFrame.fillna with 'method' is deprecated and will raise in a future version. Use obj.ffill() or obj.bfill() instead.
      print(df2.fillna(method='pad')), print('='*50)  # FutureWarning
    


```python
# 3. 데이터 처리

# 손실 데이터 처리: 손실 데이터 채우기 (2)
data=[[np.nan, 2, 0, np.nan], [3, 4, np.nan, 1], [np.nan, 5, np.nan, 2], [np.nan, 1, 2, 3]]
df = pd.DataFrame(data, columns=list('ABCD'))
print(df), print('='*50)

print(df.fillna(method='ffill')), print('='*50) # FutureWarning

val = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
print(df.fillna(value=val)), print('='*50)

# limit=1 파라미터 때문에 각 컬럼별로 최대 1개의 결측치만 채움
print(df.fillna(value=val, limit=1))

```

         A  B    C    D
    0  NaN  2  0.0  NaN
    1  3.0  4  NaN  1.0
    2  NaN  5  NaN  2.0
    3  NaN  1  2.0  3.0
    ==================================================
         A  B    C    D
    0  NaN  2  0.0  NaN
    1  3.0  4  0.0  1.0
    2  3.0  5  0.0  2.0
    3  3.0  1  2.0  3.0
    ==================================================
         A  B    C    D
    0  0.0  2  0.0  3.0
    1  3.0  4  2.0  1.0
    2  0.0  5  2.0  2.0
    3  0.0  1  2.0  3.0
    ==================================================
         A  B    C    D
    0  0.0  2  0.0  3.0
    1  3.0  4  2.0  1.0
    2  NaN  5  NaN  2.0
    3  NaN  1  2.0  3.0
    

    <ipython-input-52-1c9676f4307b>:8: FutureWarning: DataFrame.fillna with 'method' is deprecated and will raise in a future version. Use obj.ffill() or obj.bfill() instead.
      print(df.fillna(method='ffill')), print('='*50)
    


```python
# 3. 데이터 처리

# 손실 데이터 처리: 손실 데이터 채우기 (3)
df2 = pd.DataFrame({'one':[1.0, 2.0, 3.0], 'two':[4.0, 5.0, 6.0]}, index=['a','b','c'])
df2.loc['d'] = np.nan
df2['three'] = 2.0
df2.iloc[1:2, 1:2] = np.nan
print(df2), print('='*50)

# dropna() 메소드 -> 결측치가 있는 행 또는 열 제거
print(df2.dropna(axis=0)), print('='*50)

print(df2.dropna(axis=1)), print('='*50)

print(df2['two'].dropna())
```

       one  two  three
    a  1.0  4.0    2.0
    b  2.0  NaN    2.0
    c  3.0  6.0    2.0
    d  NaN  NaN    2.0
    ==================================================
       one  two  three
    a  1.0  4.0    2.0
    c  3.0  6.0    2.0
    ==================================================
       three
    a    2.0
    b    2.0
    c    2.0
    d    2.0
    ==================================================
    a    4.0
    c    6.0
    Name: two, dtype: float64
    


```python
# 3. 데이터 처리

# 손실 데이터 처리: 손실 데이터 채우기 (4)
# NaT(Not a Time): 시계열 데이터의 결측치
df = pd.DataFrame({ 'name': ['haena', 'suho', 'naeun'],
                    'hobby': ['jogging', 'reading', np.nan],
                    'born': [pd.NaT, pd.Timestamp('2001-01-01'), pd.NaT]})
                   #'born': [np.nan, pd.Timestamp('2001-01-01'), np.nan]})
print(df), print('='*50)

print(df.dropna()), print('='*50)   # 행 제거
print(df.dropna(axis='columns')), print('='*50) # 열 제거

# how='any' (default): 하나라도 결측치가 있으면 해당 행 또는 열을 제거
# how='all': 행 또는 열의 모든 값이 결측치여야 해당 행 또는 열을 제거
print(df.dropna(how='all')), print('='*50)

# 결측치가 아닌 값의 개수가 최소 2개 이상인 행만 남김
print(df.dropna(thresh=2)), print('='*50)
print(df.dropna(subset=['name', 'born'])), print('='*50)
print(df.dropna(subset=['hobby'])), print('='*50)

# 결측치가 있는 행을 제거하고 그 결과를 원본 df에 직접 반영
# inplace=True이므로 새로운 DataFrame을 반환하지 않고 None을 반환
print(df.dropna(inplace=True)), print('='*50)
print(df)
```

        name    hobby       born
    0  haena  jogging        NaT
    1   suho  reading 2001-01-01
    2  naeun      NaN        NaT
    ==================================================
       name    hobby       born
    1  suho  reading 2001-01-01
    ==================================================
        name
    0  haena
    1   suho
    2  naeun
    ==================================================
        name    hobby       born
    0  haena  jogging        NaT
    1   suho  reading 2001-01-01
    2  naeun      NaN        NaT
    ==================================================
        name    hobby       born
    0  haena  jogging        NaT
    1   suho  reading 2001-01-01
    ==================================================
       name    hobby       born
    1  suho  reading 2001-01-01
    ==================================================
        name    hobby       born
    0  haena  jogging        NaT
    1   suho  reading 2001-01-01
    ==================================================
    None
    ==================================================
       name    hobby       born
    1  suho  reading 2001-01-01
    


```python
# 3. 데이터 처리

# 손실 데이터 처리: 손실 데이터 채우기 (5)
ser = pd.Series([0, np.nan, 2, 3, 5])
print(ser), print('='*50)

# ser.replace(치환할 값, 대체할 값, inplace=False)
# 치환할 값: 스칼라, 리스트, 딕셔너리{old_value : new_value}
# 대체할 값: 스칼라, 리스트, 딕셔너리{old_value : new_value}
print(ser.replace(np.nan, 1.0)), print('='*50)

print(ser.replace({np.nan: 1, 5: 4})), print('='*50)

print(ser.replace([0, 2], 1)), print('='*50)

print(ser.replace([np.nan, 5], [1, np.nan])) # 같은 위치의 값으로 순서대로
```

    0    0.0
    1    NaN
    2    2.0
    3    3.0
    4    5.0
    dtype: float64
    ==================================================
    0    0.0
    1    1.0
    2    2.0
    3    3.0
    4    5.0
    dtype: float64
    ==================================================
    0    0.0
    1    1.0
    2    2.0
    3    3.0
    4    4.0
    dtype: float64
    ==================================================
    0    1.0
    1    NaN
    2    1.0
    3    3.0
    4    5.0
    dtype: float64
    ==================================================
    0    0.0
    1    1.0
    2    2.0
    3    3.0
    4    NaN
    dtype: float64
    


```python
# 3. 데이터 처리

# 멀티인덱스: 객체 생성 (1) from_arrays
arrays = [['IVE', 'IVE', 'AESPA', 'AESPA'],         # 레벨0
          ['Wonyoung', 'Liz', 'Winter', 'Karina']]  # 레벨1

multi_index_arrays = pd.MultiIndex.from_arrays(arrays, names=['그룹', '이름'])
print(multi_index_arrays), print('='*50)

arr = [np.array(['ha', 'ha', 'hi', 'hi', 'ho', 'ho']),        # 레벨0
       np.array(['one', 'two', 'one', 'two', 'one', 'two'])]  # 레벨1
print(arr), print('='*50)

ser = pd.Series(np.random.randn(6), index=arr)
print(ser), print('='*50)

df = pd.DataFrame(np.random.randn(6, 3), index=arr)
print(df), print('='*50)

df = pd.DataFrame(np.random.randn(3, 6), index=['A', 'B', 'C'], columns=arr)
print(df)
```

    MultiIndex([(  'IVE', 'Wonyoung'),
                (  'IVE',      'Liz'),
                ('AESPA',   'Winter'),
                ('AESPA',   'Karina')],
               names=['그룹', '이름'])
    ==================================================
    [array(['ha', 'ha', 'hi', 'hi', 'ho', 'ho'], dtype='<U2'), array(['one', 'two', 'one', 'two', 'one', 'two'], dtype='<U3')]
    ==================================================
    ha  one   -1.444844
        two   -1.264154
    hi  one    0.796867
        two   -0.507451
    ho  one   -2.273680
        two   -1.697755
    dtype: float64
    ==================================================
                   0         1         2
    ha one  0.051049  0.302227  0.716449
       two  1.030459 -1.078690 -2.461462
    hi one -0.893698 -0.594784 -0.695074
       two  0.705596  0.509806  1.932706
    ho one -0.305488 -1.606211 -0.517545
       two  0.377620  0.442207  1.119286
    ==================================================
             ha                  hi                  ho          
            one       two       one       two       one       two
    A -0.100395 -1.260629 -0.953000 -0.502841 -0.873240 -0.205307
    B  0.122393 -0.432748 -0.304985  0.358943 -0.602972  0.163765
    C -0.635100 -0.019974  0.488372  1.666418  1.638031 -1.263082
    


```python
# 3. 데이터 처리

# 멀티인덱스: 객체 생성 (2) from_frame
df = pd.DataFrame([['ha', 'one'], ['ha', 'two'], ['ho', 'one'], ['ho', 'two']],columns=['1st', '2nd'])
print(df), print('='*30)

pd.MultiIndex.from_frame(df)
```

      1st  2nd
    0  ha  one
    1  ha  two
    2  ho  one
    3  ho  two
    ==============================
    




    MultiIndex([('ha', 'one'),
                ('ha', 'two'),
                ('ho', 'one'),
                ('ho', 'two')],
               names=['1st', '2nd'])




```python
# 3. 데이터 처리

# 멀티인덱스: 객체 생성 (3) from_tuples
tuples = [('IVE', 'Wonyoung'), ('IVE', 'Liz'), ('AESPA', 'Winter'), ('AESPA', 'Karina')]
print(tuples), print('='*80)

multi_index_tuples = pd.MultiIndex.from_tuples(tuples, names=['그룹', '이름'])
print(multi_index_tuples), print('='*80)


li = [['ha', 'ha', 'hi', 'hi', 'ho', 'ho'], ['one', 'two', 'one', 'two', 'one', 'two']]
li1 = list(zip(*li))  # *는 언패킹(unpacking) 연산자, zip()은 같은 위치에 있는 요소들끼리 짝지어 튜플을 만듦
print(li1), print('='*80)

ind = pd.MultiIndex.from_tuples(li1, names=['1st', '2nd'])
print(ind), print('='*80)

ser = pd.Series(np.random.randn(6), index=ind)
print(ser)
```

    [('IVE', 'Wonyoung'), ('IVE', 'Liz'), ('AESPA', 'Winter'), ('AESPA', 'Karina')]
    ================================================================================
    MultiIndex([(  'IVE', 'Wonyoung'),
                (  'IVE',      'Liz'),
                ('AESPA',   'Winter'),
                ('AESPA',   'Karina')],
               names=['그룹', '이름'])
    ================================================================================
    [('ha', 'one'), ('ha', 'two'), ('hi', 'one'), ('hi', 'two'), ('ho', 'one'), ('ho', 'two')]
    ================================================================================
    MultiIndex([('ha', 'one'),
                ('ha', 'two'),
                ('hi', 'one'),
                ('hi', 'two'),
                ('ho', 'one'),
                ('ho', 'two')],
               names=['1st', '2nd'])
    ================================================================================
    1st  2nd
    ha   one    0.436787
         two   -0.145452
    hi   one   -0.935653
         two    1.410856
    ho   one   -0.607751
         two    1.115869
    dtype: float64
    


```python
# 3. 데이터 처리

# 멀티인덱스: 객체 생성 (4) from_product
groups = ['IVE', 'AESPA']                           # 레벨0
names = ['Wonyoung', 'Liz', 'Winter', 'Karina']     # 레벨1
multi_index_product = pd.MultiIndex.from_product([groups, names], names=['그룹', '이름'])
print(multi_index_product), print('='*80)

iter = [['ha', 'hi', 'ho'], ['one', 'two']]
print(pd.MultiIndex.from_product(iter, names=['1st', '2nd']))
```

    MultiIndex([(  'IVE', 'Wonyoung'),
                (  'IVE',      'Liz'),
                (  'IVE',   'Winter'),
                (  'IVE',   'Karina'),
                ('AESPA', 'Wonyoung'),
                ('AESPA',      'Liz'),
                ('AESPA',   'Winter'),
                ('AESPA',   'Karina')],
               names=['그룹', '이름'])
    ================================================================================
    MultiIndex([('ha', 'one'),
                ('ha', 'two'),
                ('hi', 'one'),
                ('hi', 'two'),
                ('ho', 'one'),
                ('ho', 'two')],
               names=['1st', '2nd'])
    


```python
# 3. 데이터 처리

# 멀티인덱스: 인덱싱 (1)
arr = [np.array(['ha', 'ha', 'hi', 'hi', 'ho', 'ho']),        # 레벨0
       np.array(['one', 'two', 'one', 'two', 'one', 'two'])]  # 레벨1
print(arr), print('='*50)

df = pd.DataFrame(np.random.randn(3, 6), index=['A', 'B', 'C'], columns=arr)
print(df), print('='*50)

print(df['ha']), print('='*50)

print(df['ha']['one']), print('='*50)


ser = pd.Series(np.random.randn(6), index=arr)
print(ser), print('='*50)

print(ser.reindex([('ho', 'one'), ('ha', 'two')]))
```

    [array(['ha', 'ha', 'hi', 'hi', 'ho', 'ho'], dtype='<U2'), array(['one', 'two', 'one', 'two', 'one', 'two'], dtype='<U3')]
    ==================================================
             ha                  hi                  ho          
            one       two       one       two       one       two
    A  0.486212 -1.500578  1.338625  1.064739 -0.861161 -0.048550
    B  1.014602  0.576160  0.718637 -1.196733 -2.920774  0.246061
    C  0.488492 -0.951324  1.618496 -1.112683  1.294821  0.194799
    ==================================================
            one       two
    A  0.486212 -1.500578
    B  1.014602  0.576160
    C  0.488492 -0.951324
    ==================================================
    A    0.486212
    B    1.014602
    C    0.488492
    Name: one, dtype: float64
    ==================================================
    ha  one   -0.770134
        two   -0.846643
    hi  one    0.691360
        two   -1.568186
    ho  one   -0.609667
        two    0.790820
    dtype: float64
    ==================================================
    ho  one   -0.609667
    ha  two   -0.846643
    dtype: float64
    


```python
# 3. 데이터 처리

# 멀티인덱스: 인덱싱 (2)
df = pd.DataFrame(np.random.randn(3, 6), index=['A', 'B', 'C'], columns=arr)
print(df), print('='*50)

df = df.T
print(df), print('='*50)

print(df.loc[('ha', 'two')]), print('='*50)
print(df.loc[('ha', 'two'), 'A']), print('='*50)
print(df.loc['ha']), print('='*50)
print(df.loc['ha':'hi']), print('='*50)
print(df.loc[('hi', 'two'):('ho', 'one')]), print('='*50)
print(df.loc[('hi', 'two'):'ho']), print('='*50)
print(df.loc[[('ha', 'two'), ('ho', 'one')]])
```

             ha                  hi                  ho          
            one       two       one       two       one       two
    A  0.186847 -1.184557  1.250534 -1.283592  0.009752 -0.835926
    B  0.195396 -1.376633  0.677585 -0.154793 -0.846654 -0.680803
    C  0.899658  1.289674  0.820775  1.070275  0.103825  0.091563
    ==================================================
                   A         B         C
    ha one  0.186847  0.195396  0.899658
       two -1.184557 -1.376633  1.289674
    hi one  1.250534  0.677585  0.820775
       two -1.283592 -0.154793  1.070275
    ho one  0.009752 -0.846654  0.103825
       two -0.835926 -0.680803  0.091563
    ==================================================
    A   -1.184557
    B   -1.376633
    C    1.289674
    Name: (ha, two), dtype: float64
    ==================================================
    -1.1845568498145098
    ==================================================
                A         B         C
    one  0.186847  0.195396  0.899658
    two -1.184557 -1.376633  1.289674
    ==================================================
                   A         B         C
    ha one  0.186847  0.195396  0.899658
       two -1.184557 -1.376633  1.289674
    hi one  1.250534  0.677585  0.820775
       two -1.283592 -0.154793  1.070275
    ==================================================
                   A         B         C
    hi two -1.283592 -0.154793  1.070275
    ho one  0.009752 -0.846654  0.103825
    ==================================================
                   A         B         C
    hi two -1.283592 -0.154793  1.070275
    ho one  0.009752 -0.846654  0.103825
       two -0.835926 -0.680803  0.091563
    ==================================================
                   A         B         C
    ha two -1.184557 -1.376633  1.289674
    ho one  0.009752 -0.846654  0.103825
    


```python
# 3. 데이터 처리

# 멀티인덱스: 순서 정렬
li = [['ha', 'ha', 'hi', 'hi', 'ho', 'ho'], ['one', 'two', 'one', 'two', 'one', 'two']]
li1 = list(zip(*li))  # *는 언패킹(unpacking) 연산자, zip()은 같은 위치에 있는 요소들끼리 짝지어 튜플을 만듦
print(li1), print('='*50)

np.random.shuffle(li1)
print(li1), print('='*50)

ser = pd.Series(np.random.randn(6), index=pd.MultiIndex.from_tuples(li1))
print(ser), print('='*50)

print(ser.sort_index()), print('='*50)

print(ser.sort_index(level=0)), print('='*50)
print(ser.sort_index(level=1)), print('='*50)

ser.index.set_names(['1st', '2nd'], inplace=True)
print(ser.sort_index(level='1st')), print('='*50)
print(ser.sort_index(level='2nd')), print('='*50)

print(df), print('='*50)
print(df.T.sort_index(axis=1, level=1))
```

    [('ha', 'one'), ('ha', 'two'), ('hi', 'one'), ('hi', 'two'), ('ho', 'one'), ('ho', 'two')]
    ==================================================
    [('hi', 'two'), ('ha', 'two'), ('ho', 'one'), ('hi', 'one'), ('ha', 'one'), ('ho', 'two')]
    ==================================================
    hi  two   -1.532188
    ha  two   -0.173921
    ho  one   -0.446655
    hi  one   -0.932865
    ha  one    0.830245
    ho  two   -0.945872
    dtype: float64
    ==================================================
    ha  one    0.830245
        two   -0.173921
    hi  one   -0.932865
        two   -1.532188
    ho  one   -0.446655
        two   -0.945872
    dtype: float64
    ==================================================
    ha  one    0.830245
        two   -0.173921
    hi  one   -0.932865
        two   -1.532188
    ho  one   -0.446655
        two   -0.945872
    dtype: float64
    ==================================================
    ha  one    0.830245
    hi  one   -0.932865
    ho  one   -0.446655
    ha  two   -0.173921
    hi  two   -1.532188
    ho  two   -0.945872
    dtype: float64
    ==================================================
    1st  2nd
    ha   one    0.830245
         two   -0.173921
    hi   one   -0.932865
         two   -1.532188
    ho   one   -0.446655
         two   -0.945872
    dtype: float64
    ==================================================
    1st  2nd
    ha   one    0.830245
    hi   one   -0.932865
    ho   one   -0.446655
    ha   two   -0.173921
    hi   two   -1.532188
    ho   two   -0.945872
    dtype: float64
    ==================================================
             ha                  hi                  ho          
            one       two       one       two       one       two
    A  0.486212 -1.500578  1.338625  1.064739 -0.861161 -0.048550
    B  1.014602  0.576160  0.718637 -1.196733 -2.920774  0.246061
    C  0.488492 -0.951324  1.618496 -1.112683  1.294821  0.194799
    ==================================================
                   A         B         C
    ha one  0.486212  1.014602  0.488492
       two -1.500578  0.576160 -0.951324
    hi one  1.338625  0.718637  1.618496
       two  1.064739 -1.196733 -1.112683
    ho one -0.861161 -2.920774  1.294821
       two -0.048550  0.246061  0.194799
    


```python
# 4. 데이터 타입과 입출력

# 텍스트 파일: CSV 파일
data = {'name': ['haena', 'naeun', 'una', 'bum', 'suho'],
        'age': [30, 27, 28, 23, 18],
        'address': ['dogok', 'suwon', 'mapo', 'ilsan', 'yeoyi'],
        'grade': ['A', 'B', 'C', 'B', 'A'],
        'score': [100, 88, 73, 83, 95]}

df = pd.DataFrame(data, columns=['name', 'age', 'address', 'score', 'grade'])
print(df), print('='*40)

df.to_csv('./student_grade.csv')
df.to_csv('./student_grade_noIndex.csv', index=False)

# OS가 windows일 경우,
!type student_grade.csv
!more student_grade.csv

# OS가 UNIX, macOS일 경우,
!cat student_grade.csv
```

        name  age address  score grade
    0  haena   30   dogok    100     A
    1  naeun   27   suwon     88     B
    2    una   28    mapo     73     C
    3    bum   23   ilsan     83     B
    4   suho   18   yeoyi     95     A
    ========================================
    /bin/bash: line 1: type: student_grade.csv: not found
    ,name,age,address,score,grade
    0,haena,30,dogok,100,A
    1,naeun,27,suwon,88,B
    2,una,28,mapo,73,C
    3,bum,23,ilsan,83,B
    4,suho,18,yeoyi,95,A
    ,name,age,address,score,grade
    0,haena,30,dogok,100,A
    1,naeun,27,suwon,88,B
    2,una,28,mapo,73,C
    3,bum,23,ilsan,83,B
    4,suho,18,yeoyi,95,A
    


```python
# 4. 데이터 타입과 입출력

# 텍스트 파일: CSV 파일
df_basic = pd.read_csv('student_grade.csv')
print(df_basic), print('='*50)

df_basic2 = pd.read_csv('student_grade_noIndex.csv')
print(df_basic2), print('='*50)

df_sep = pd.read_csv('student_grade.csv', sep=',')
print(df_sep)
```

       Unnamed: 0   name  age address  score grade
    0           0  haena   30   dogok    100     A
    1           1  naeun   27   suwon     88     B
    2           2    una   28    mapo     73     C
    3           3    bum   23   ilsan     83     B
    4           4   suho   18   yeoyi     95     A
    ==================================================
        name  age address  score grade
    0  haena   30   dogok    100     A
    1  naeun   27   suwon     88     B
    2    una   28    mapo     73     C
    3    bum   23   ilsan     83     B
    4   suho   18   yeoyi     95     A
    ==================================================
       Unnamed: 0   name  age address  score grade
    0           0  haena   30   dogok    100     A
    1           1  naeun   27   suwon     88     B
    2           2    una   28    mapo     73     C
    3           3    bum   23   ilsan     83     B
    4           4   suho   18   yeoyi     95     A
    


```python
# 4. 데이터 타입과 입출력

# 텍스트 파일: CSV 파일
# Unnamed: 0 -> 인덱스(index) 정보가 첫 번째 열로 저장
# Unnamed: 0 열 제거 방법1
df_slicing = df1.iloc[0:, 1:]
print(df_slicing), print('='*40)

# Unnamed: 0 열 제거 방법2: 파일을 읽어면서 첫 번째 열을 인덱스로 사용
df_index_col = pd.read_csv('student_grade.csv', index_col=0)
print(df_index_col), print('='*40)

df_index_col2 = pd.read_csv('student_grade.csv', index_col=['name'])
print(df_index_col2)
```

        name  age address  score grade
    0  haena   30   dogok    100     A
    1  naeun   27   suwon     88     B
    2    una   28    mapo     73     C
    3    bum   23   ilsan     83     B
    4   suho   18   yeoyi     95     A
    ========================================
        name  age address  score grade
    0  haena   30   dogok    100     A
    1  naeun   27   suwon     88     B
    2    una   28    mapo     73     C
    3    bum   23   ilsan     83     B
    4   suho   18   yeoyi     95     A
    ========================================
           Unnamed: 0  age address  score grade
    name                                       
    haena           0   30   dogok    100     A
    naeun           1   27   suwon     88     B
    una             2   28    mapo     73     C
    bum             3   23   ilsan     83     B
    suho            4   18   yeoyi     95     A
    


```python
# 4. 데이터 타입과 입출력

# 텍스트 파일: CSV 파일
# usecols에 필요한 컬럼 인덱스만 지정
df_usecols = pd.read_csv('student_grade.csv',
                         usecols=['name', 'age', 'address', 'grade', 'score'])
print(df_usecols), print('='*40)

df_usecols2 = pd.read_csv('student_grade.csv', index_col=['name'],
                         usecols=['name', 'age', 'address', 'grade', 'score'])
print(df_usecols2), print('='*40)

# 컬럼이 아주 많을 때, 슬라이싱이나 아래의 drop으로 Unnamed: 0 열 제거
df_drop = pd.read_csv('student_grade.csv').drop(columns=['Unnamed: 0'])
print(df_drop)
```

        name  age address  score grade
    0  haena   30   dogok    100     A
    1  naeun   27   suwon     88     B
    2    una   28    mapo     73     C
    3    bum   23   ilsan     83     B
    4   suho   18   yeoyi     95     A
    ========================================
           age address  score grade
    name                           
    haena   30   dogok    100     A
    naeun   27   suwon     88     B
    una     28    mapo     73     C
    bum     23   ilsan     83     B
    suho    18   yeoyi     95     A
    ========================================
        name  age address  score grade
    0  haena   30   dogok    100     A
    1  naeun   27   suwon     88     B
    2    una   28    mapo     73     C
    3    bum   23   ilsan     83     B
    4   suho   18   yeoyi     95     A
    


```python
# 4. 데이터 타입과 입출력

# 텍스트 파일: CSV 파일

# header = None -> 컬럼 이름이 있는 상태에서 None으로 두면,
#                  컬럼 이름이 0번째 행으로 이동 (주의!!!)
df_header = pd.read_csv('student_grade.csv', header=None)
print(df_header), print('='*40)

# nrows (읽을 행의 수 제한)
df_nrows = pd.read_csv('student_grade.csv', nrows=3)
print(df_nrows), print('='*40)

df_nrows2 = pd.read_csv('student_grade.csv', header=None, nrows=3, index_col=0)
print(df_nrows2), print('='*40)

df6 = pd.read_csv('student_grade.csv',
                  names=['No','name','age','address','score','grade'], nrows=3)
print(df6)
```

         0      1    2        3      4      5
    0  NaN   name  age  address  score  grade
    1  0.0  haena   30    dogok    100      A
    2  1.0  naeun   27    suwon     88      B
    3  2.0    una   28     mapo     73      C
    4  3.0    bum   23    ilsan     83      B
    5  4.0   suho   18    yeoyi     95      A
    ========================================
       Unnamed: 0   name  age address  score grade
    0           0  haena   30   dogok    100     A
    1           1  naeun   27   suwon     88     B
    2           2    una   28    mapo     73     C
    ========================================
             1    2        3      4      5
    0                                     
    NaN   name  age  address  score  grade
    0.0  haena   30    dogok    100      A
    1.0  naeun   27    suwon     88      B
    ========================================
        No   name  age  address  score  grade
    0  NaN   name  age  address  score  grade
    1  0.0  haena   30    dogok    100      A
    2  1.0  naeun   27    suwon     88      B
    


```python
# 4. 데이터 타입과 입출력

# 텍스트 파일: CSV 파일
# na_values: 결측치로 인식할 값 지정
# 'address' 컬럼에서는 값 'mapo'와 'NA'를 결측치로 간주
# 'score' 컬럼에서는 값 83을 결측치로 간주
to_na = {'address': ['mapo', 'NA'], 'score': [83]}
df_na_values = pd.read_csv('student_grade.csv', na_values=to_na)
print(df_na_values), print('='*40)

# skiprows: 특정 행 건너뛰기
df_skiprows = pd.read_csv('student_grade.csv', skiprows=3)
print(df_skiprows)
```

       Unnamed: 0   name  age address  score grade
    0           0  haena   30   dogok  100.0     A
    1           1  naeun   27   suwon   88.0     B
    2           2    una   28     NaN   73.0     C
    3           3    bum   23   ilsan    NaN     B
    4           4   suho   18   yeoyi   95.0     A
    ========================================
       2   una  28   mapo  73  C
    0  3   bum  23  ilsan  83  B
    1  4  suho  18  yeoyi  95  A
    ========================================
    




    (None, None)




```python
# 4. 데이터 타입과 입출력

# 텍스트 파일: JSON 파일
# filePath=None 으로 설정하면, JSON 형식의 문자열을 반환
dfj = pd.DataFrame([['a', 'b'], ['c', 'd']],
                   index=['row1', 'row2'], columns=['col1', 'col2'])
print(dfj), print('='*40)
dfj.to_json()
print(dfj.to_json())

# filePath 지정하면 파일이 저장됨
dfj.to_json('my_data.json')
```

         col1 col2
    row1    a    b
    row2    c    d
    ========================================
    {"col1":{"row1":"a","row2":"c"},"col2":{"row1":"b","row2":"d"}}
    


```python
# 'columns' (기본값 for DataFrame):
# 컬럼 이름이 키(key)가 되고, 각 컬럼의 값이 배열(리스트)이 됨
# {column -> {index -> values}}와 같은 딕셔너리형이 됨
dfj.to_json(orient='columns')
```




    '{"col1":{"row1":"a","row2":"c"},"col2":{"row1":"b","row2":"d"}}'




```python
# 'index': 인덱스(index)가 키(key)가 되고,
# 각 인덱스에 해당하는 행의 값이 객체(딕셔너리)가 됨
# {index -> {column -> value}}와 같은 딕셔너리형이 됨
dfj.to_json(orient='index')
```




    '{"row1":{"col1":"a","col2":"b"},"row2":{"col1":"c","col2":"d"}}'




```python
# 'split': 딕셔너리에 columns, index, data 키가 포함됨
dfj.to_json(orient='split')
```




    '{"columns":["col1","col2"],"index":["row1","row2"],"data":[["a","b"],["c","d"]]}'




```python
# 'records': 각 행이 하나의 JSON 객체가 되고,
# 이 객체들이 배열(리스트) 안에 포함됨
# 이 형식이 웹 API에서 가장 흔하게 사용됨
dfj.to_json(orient='records')
```




    '[{"col1":"a","col2":"b"},{"col1":"c","col2":"d"}]'




```python
# 'values': DataFrame/Series의 값들만 포함된 중첩된 배열(리스트)이 됨
# 인덱스나 컬럼 이름은 포함되지 않음
dfj.to_json(orient='values')
```




    '[["a","b"],["c","d"]]'




```python
# 'table': JSON Table Schema 형식을 따름
dfj.to_json(orient='table')
```




    '{"schema":{"fields":[{"name":"index","type":"string"},{"name":"col1","type":"string"},{"name":"col2","type":"string"}],"primaryKey":["index"],"pandas_version":"1.4.0"},"data":[{"index":"row1","col1":"a","col2":"b"},{"index":"row2","col1":"c","col2":"d"}]}'




```python
# 4. 데이터 타입과 입출력

# 텍스트 파일: JSON 파일
df = pd.DataFrame({
    'col1': [1, 2, 3],
    'col2': ['A', 'B', 'C'],
    'date_col': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])
}, index=['idx1', 'idx2', 'idx3'])

print(f"원본 DataFrame: \n{df}"), print("-" * 30)

# 1. 기본 저장 (orient='columns'이 DataFrame의 기본값)
# 파일로 저장하지 않고 문자열로 반환
json_str_columns = df.to_json() # indent=None (기본값): 들여쓰기 없이 한 줄로 출력
#json_str_columns = df.to_json(indent=2)
print(f"orient='columns' (기본값): \n{json_str_columns}"), print("-" * 30)

# 2. orient='records' (웹 API에서 흔히 사용)
json_str_records = df.to_json(orient='records', indent=2) # indent로 가독성 높이기
print(f"orient='records' (indent=2): \n{json_str_records}"), print("-" * 30)

# 3. orient='index'
json_str_index = df.to_json(orient='index', indent=2)
print(f"orient='index' (indent=2): \n{json_str_index}"), print("-" * 30)

# 4. orient='values'
json_str_values = df.to_json(orient='values', indent=2)
print(f"orient='values' (indent=2): \n{json_str_values}"), print("-" * 30)

# 5. orient='split'
json_str_split = df.to_json(orient='split', indent=2)
print(f"orient='split' (indent=2): \n{json_str_split}"), print("-" * 30)

# 6. orient='table'
json_str_table = df.to_json(orient='table', indent=2)
print(f"orient='table' (indent=2): \n{json_str_table}")

```

    원본 DataFrame: 
          col1 col2   date_col
    idx1     1    A 2023-01-01
    idx2     2    B 2023-01-02
    idx3     3    C 2023-01-03
    ------------------------------
    orient='columns' (기본값): 
    {"col1":{"idx1":1,"idx2":2,"idx3":3},"col2":{"idx1":"A","idx2":"B","idx3":"C"},"date_col":{"idx1":1672531200000,"idx2":1672617600000,"idx3":1672704000000}}
    ------------------------------
    orient='records' (indent=2): 
    [
        {
            "col1":1,
            "col2":"A",
            "date_col":1672531200000
        },
        {
            "col1":2,
            "col2":"B",
            "date_col":1672617600000
        },
        {
            "col1":3,
            "col2":"C",
            "date_col":1672704000000
        }
    ]
    ------------------------------
    orient='index' (indent=2): 
    {
      "idx1":{
        "col1":1,
        "col2":"A",
        "date_col":1672531200000
      },
      "idx2":{
        "col1":2,
        "col2":"B",
        "date_col":1672617600000
      },
      "idx3":{
        "col1":3,
        "col2":"C",
        "date_col":1672704000000
      }
    }
    ------------------------------
    orient='values' (indent=2): 
    [
      [
        1,
        "A",
        1672531200000
      ],
      [
        2,
        "B",
        1672617600000
      ],
      [
        3,
        "C",
        1672704000000
      ]
    ]
    ------------------------------
    orient='split' (indent=2): 
    {
      "columns":[
        "col1",
        "col2",
        "date_col"
      ],
      "index":[
        "idx1",
        "idx2",
        "idx3"
      ],
      "data":[
        [
          1,
          "A",
          1672531200000
        ],
        [
          2,
          "B",
          1672617600000
        ],
        [
          3,
          "C",
          1672704000000
        ]
      ]
    }
    ------------------------------
    orient='table' (indent=2): 
    {
      "schema":{
        "fields":[
          {
            "name":"index",
            "type":"string"
          },
          {
            "name":"col1",
            "type":"integer"
          },
          {
            "name":"col2",
            "type":"string"
          },
          {
            "name":"date_col",
            "type":"datetime"
          }
        ],
        "primaryKey":[
          "index"
        ],
        "pandas_version":"1.4.0"
      },
      "data":[
        {
          "index":"idx1",
          "col1":1,
          "col2":"A",
          "date_col":"2023-01-01T00:00:00.000"
        },
        {
          "index":"idx2",
          "col1":2,
          "col2":"B",
          "date_col":"2023-01-02T00:00:00.000"
        },
        {
          "index":"idx3",
          "col1":3,
          "col2":"C",
          "date_col":"2023-01-03T00:00:00.000"
        }
      ]
    }
    


```python
# 4. 데이터 타입과 입출력

# 텍스트 파일: JSON 파일
data = {'name': ['haena', 'naeun', 'una', 'bum', 'suho'],
        'age': [30, 27, 28, 23, 18],
        'address': ['dogok', 'suwon', 'mapo', 'ilsan', 'yeoyi'],
        'grade': ['A', 'B', 'C', 'B', 'A'],
        'score': [100, 88, 73, 83, 95]}

df = pd.DataFrame(data, columns=['name', 'age', 'address', 'score', 'grade'])
df.to_json('student_grade.json', indent=2)

pd.read_json('student_grade.json')
```





  <div id="df-3f47d87f-2a5f-4b2b-a35e-fed8109b70ec" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>age</th>
      <th>address</th>
      <th>score</th>
      <th>grade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>haena</td>
      <td>30</td>
      <td>dogok</td>
      <td>100</td>
      <td>A</td>
    </tr>
    <tr>
      <th>1</th>
      <td>naeun</td>
      <td>27</td>
      <td>suwon</td>
      <td>88</td>
      <td>B</td>
    </tr>
    <tr>
      <th>2</th>
      <td>una</td>
      <td>28</td>
      <td>mapo</td>
      <td>73</td>
      <td>C</td>
    </tr>
    <tr>
      <th>3</th>
      <td>bum</td>
      <td>23</td>
      <td>ilsan</td>
      <td>83</td>
      <td>B</td>
    </tr>
    <tr>
      <th>4</th>
      <td>suho</td>
      <td>18</td>
      <td>yeoyi</td>
      <td>95</td>
      <td>A</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-3f47d87f-2a5f-4b2b-a35e-fed8109b70ec')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-3f47d87f-2a5f-4b2b-a35e-fed8109b70ec button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-3f47d87f-2a5f-4b2b-a35e-fed8109b70ec');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


    <div id="df-b87d02f5-55c5-4a97-b7d1-1e1b7340a1c8">
      <button class="colab-df-quickchart" onclick="quickchart('df-b87d02f5-55c5-4a97-b7d1-1e1b7340a1c8')"
                title="Suggest charts"
                style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
      </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

      <script>
        async function quickchart(key) {
          const quickchartButtonEl =
            document.querySelector('#' + key + ' button');
          quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
          quickchartButtonEl.classList.add('colab-df-spinner');
          try {
            const charts = await google.colab.kernel.invokeFunction(
                'suggestCharts', [key], {});
          } catch (error) {
            console.error('Error during call to suggestCharts:', error);
          }
          quickchartButtonEl.classList.remove('colab-df-spinner');
          quickchartButtonEl.classList.add('colab-df-quickchart-complete');
        }
        (() => {
          let quickchartButtonEl =
            document.querySelector('#df-b87d02f5-55c5-4a97-b7d1-1e1b7340a1c8 button');
          quickchartButtonEl.style.display =
            google.colab.kernel.accessAllowed ? 'block' : 'none';
        })();
      </script>
    </div>

    </div>
  </div>





```python
# 4. 데이터 타입과 입출력

# 텍스트 파일: JSON 파일
df = pd.DataFrame({'ha': [1, 2, 3, 4],
                   'hi': ['a', 'b', 'c', 'd'],
                   'ho': pd.date_range('2021-09-01', freq='d', periods=4),
                   'hu': pd.Categorical(['a', 'b', 'c', 'd'])},
index = pd.Index(range(4), name='ind'))
print(df), print('='*40)

print(df.dtypes), print('='*40)

df.to_json('hello.json', orient='table')

dfj = pd.read_json('hello.json', orient='table')
# 구조가 달라서 읽을 때 오류 발생
#dfj = pd.read_json('hello.json', orient='index')
print(dfj), print('='*40)
print(dfj.dtypes)
```

         ha hi         ho hu
    ind                     
    0     1  a 2021-09-01  a
    1     2  b 2021-09-02  b
    2     3  c 2021-09-03  c
    3     4  d 2021-09-04  d
    ========================================
    ha             int64
    hi            object
    ho    datetime64[ns]
    hu          category
    dtype: object
    ========================================
         ha hi         ho hu
    ind                     
    0     1  a 2021-09-01  a
    1     2  b 2021-09-02  b
    2     3  c 2021-09-03  c
    3     4  d 2021-09-04  d
    ========================================
    ha             int64
    hi            object
    ho    datetime64[ns]
    hu          category
    dtype: object
    


```python
# 4. 데이터 타입과 입출력

# 이진 데이터: 엑셀 파일
df = pd.read_excel('shoppingcenter.xlsx')
print(df), print('='*100)

df1 = pd.read_excel('shoppingcenter.xlsx', sheet_name='2018년 하반기 업력현황')
print(df1), print('='*100)

df2 = pd.read_excel('shoppingcenter.xlsx', sheet_name=None)
print(df2)

df.to_excel('shoppingcenter_sheet1.xlsx', sheet_name='Sheet1')
```

              광역시도   시군구     업종대분류         업종중분류  1년미만  1~2년  2~3년  3~5년  5년 이상
    0        서울특별시   종로구  관광/여가/오락      연극/영화/극장   1.0   2.0  42.0  39.0   16.0
    1        서울특별시   종로구  관광/여가/오락         전시/관람   0.0   6.0  18.0  35.0   44.0
    2        서울특별시   종로구  관광/여가/오락  PC/오락/당구/볼링등   0.0   5.0  12.0  86.0   20.0
    3        서울특별시   종로구  관광/여가/오락    경마/경륜/성인오락   0.0   1.0   1.0   1.0    2.0
    4        서울특별시   종로구  관광/여가/오락        스포츠/운동   0.0   0.0   1.0   0.0    0.0
    ...        ...   ...       ...           ...   ...   ...   ...   ...    ...
    20620  제주특별자치도  서귀포시     학문/교육     학원-예능취미체육   5.0   1.0   1.0   0.0   22.0
    20621  제주특별자치도  서귀포시     학문/교육     학원-보습교습입시   6.0   2.0   3.0  47.0   18.0
    20622  제주특별자치도  서귀포시     학문/교육          학원기타   8.0   1.0  17.0  53.0   28.0
    20623  제주특별자치도  서귀포시     학문/교육          유아교육   0.0   1.0   7.0   5.0  115.0
    20624  제주특별자치도  서귀포시     학문/교육        특수교육기관   0.0   1.0   0.0   0.0    3.0
    
    [20625 rows x 9 columns]
    ====================================================================================================
              광역시도   시군구     업종대분류         업종중분류  1년미만  1~2년  2~3년  3~5년  5년 이상
    0        서울특별시   종로구  관광/여가/오락      연극/영화/극장     1     1    16    64     19
    1        서울특별시   종로구  관광/여가/오락         전시/관람     0     0    13    45     46
    2        서울특별시   종로구  관광/여가/오락  PC/오락/당구/볼링등     0     3     9    68     43
    3        서울특별시   종로구  관광/여가/오락    경마/경륜/성인오락     0     1     2     0      3
    4        서울특별시   종로구  관광/여가/오락        스포츠/운동     0     0     1     0      0
    ...        ...   ...       ...           ...   ...   ...   ...   ...    ...
    20620  제주특별자치도  서귀포시     학문/교육     학원-예능취미체육     2     3     1     1     22
    20621  제주특별자치도  서귀포시     학문/교육     학원-보습교습입시     3     4     3    47     18
    20622  제주특별자치도  서귀포시     학문/교육          학원기타     4     4     6    64     28
    20623  제주특별자치도  서귀포시     학문/교육          유아교육     0     0     3     9    115
    20624  제주특별자치도  서귀포시     학문/교육        특수교육기관     0     0     1     0      3
    
    [20625 rows x 9 columns]
    ====================================================================================================
    {'2018년 상반기 업력현황':           광역시도   시군구     업종대분류         업종중분류  1년미만  1~2년  2~3년  3~5년  5년 이상
    0        서울특별시   종로구  관광/여가/오락      연극/영화/극장   1.0   2.0  42.0  39.0   16.0
    1        서울특별시   종로구  관광/여가/오락         전시/관람   0.0   6.0  18.0  35.0   44.0
    2        서울특별시   종로구  관광/여가/오락  PC/오락/당구/볼링등   0.0   5.0  12.0  86.0   20.0
    3        서울특별시   종로구  관광/여가/오락    경마/경륜/성인오락   0.0   1.0   1.0   1.0    2.0
    4        서울특별시   종로구  관광/여가/오락        스포츠/운동   0.0   0.0   1.0   0.0    0.0
    ...        ...   ...       ...           ...   ...   ...   ...   ...    ...
    20620  제주특별자치도  서귀포시     학문/교육     학원-예능취미체육   5.0   1.0   1.0   0.0   22.0
    20621  제주특별자치도  서귀포시     학문/교육     학원-보습교습입시   6.0   2.0   3.0  47.0   18.0
    20622  제주특별자치도  서귀포시     학문/교육          학원기타   8.0   1.0  17.0  53.0   28.0
    20623  제주특별자치도  서귀포시     학문/교육          유아교육   0.0   1.0   7.0   5.0  115.0
    20624  제주특별자치도  서귀포시     학문/교육        특수교육기관   0.0   1.0   0.0   0.0    3.0
    
    [20625 rows x 9 columns], '2018년 하반기 업력현황':           광역시도   시군구     업종대분류         업종중분류  1년미만  1~2년  2~3년  3~5년  5년 이상
    0        서울특별시   종로구  관광/여가/오락      연극/영화/극장     1     1    16    64     19
    1        서울특별시   종로구  관광/여가/오락         전시/관람     0     0    13    45     46
    2        서울특별시   종로구  관광/여가/오락  PC/오락/당구/볼링등     0     3     9    68     43
    3        서울특별시   종로구  관광/여가/오락    경마/경륜/성인오락     0     1     2     0      3
    4        서울특별시   종로구  관광/여가/오락        스포츠/운동     0     0     1     0      0
    ...        ...   ...       ...           ...   ...   ...   ...   ...    ...
    20620  제주특별자치도  서귀포시     학문/교육     학원-예능취미체육     2     3     1     1     22
    20621  제주특별자치도  서귀포시     학문/교육     학원-보습교습입시     3     4     3    47     18
    20622  제주특별자치도  서귀포시     학문/교육          학원기타     4     4     6    64     28
    20623  제주특별자치도  서귀포시     학문/교육          유아교육     0     0     3     9    115
    20624  제주특별자치도  서귀포시     학문/교육        특수교육기관     0     0     1     0      3
    
    [20625 rows x 9 columns]}
    


```python
# 4. 데이터 타입과 입출력

# 이진 데이터: HDF5 파일
import tensorflow as tf
from tensorflow import keras
import numpy as np

# 예시 모델 생성 및 학습
model = keras.Sequential([
    keras.layers.Dense(10, activation='relu', input_shape=(5,)),
    keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
x = np.random.rand(100, 5)
y = np.random.rand(100, 1)
model.fit(x, y, epochs=2)

# 모델 전체를 .h5 파일로 저장
model.save('my_model.h5')

# 저장된 모델을 다시 로드
loaded_model = keras.models.load_model('my_model.h5')

# 결과 확인 (동일한 입력에 대해 동일한 예측)
print(np.allclose(model.predict(x), loaded_model.predict(x)))
```


```python
# 4. 데이터 타입과 입출력

# 이진 데이터: .keras 파일
model.save('my_model.keras')
loaded_model = tf.keras.models.load_model('my_model.keras')
```


```python
# 4. 데이터 타입과 입출력

# 이진 데이터: .pt, .pth 파일
import torch
import torch.nn as nn

# 예시 모델 정의
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(4, 2)
    def forward(self, x):
        return self.fc(x)

model = SimpleModel()

# 모델 전체 저장
torch.save(model, 'model.pt')  # 또는 'model.pth'

# 모델 전체 로딩
loaded_model = torch.load('model.pt')
loaded_model.eval()  # 추론 시에는 eval() 호출

# 파라미터만 저장
torch.save(model.state_dict(), 'model_state_dict.pth')

# 파라미터만 로딩
model2 = SimpleModel()  # 동일한 구조의 모델 인스턴스 필요
model2.load_state_dict(torch.load('model_state_dict.pth'))
model2.eval()  # 추론 시에는 eval() 호출
```


```python

```


```python

```

## 판다스 고급

### 데이터 가공


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
frames = [df1, df2, df3]

# axis 매개변수가 지정되지 않았으므로 기본값인 axis=0
# (행을 기준으로 위아래로 이어 붙이기)이 적용
result = pd.concat(frames)
result1 = pd.concat(frames, keys=['x', 'y', 'z']) # 멀티 인덱스

print(result), print('='*50)
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
df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                    'B': ['B0', 'B1', 'B2'],
                    'C': ['C0', 'C1', 'C2']}, index=[0, 1, 2])

df4 = pd.DataFrame({'B': ['B2', 'B6', 'B7'],
                    'C': ['C2', 'C6', 'C7'],
                    'E': ['E2', 'E6', 'E7']}, index=[2, 6, 7])

# axis=1: 데이터를 열(column)을 기준으로 결합. 즉, 두 옆으로 이어 붙임
# sort=False: 원래 DataFrame의 열 순서로 정렬하지 않고 유지
# 열을 기준으로 결합할 때는 두 DataFrame의 인덱스를 기준으로 align
# 인덱스가 일치하지 않는 부분은 NaN으로 채워짐
result = pd.concat([df1, df4], axis=1, sort=False)
print(result), print('='*50)

# join='inner': 두 DataFrame에서 모두 존재하는 인덱스(또는 열)의 교집합만 결과에 포함시킴
result1 = pd.concat([df1, df4], axis=1, join='inner')
print(result1), print('='*50)

# join='outer' (기본값): 두 DataFrame의 모든 인덱스(또는 열)의 합집합을 결과에 포함시킴
# 일치하지 않는 부분은 NaN으로 채움
result2 = pd.concat([df1, df4], axis=1, join='outer')
print(result2)
```

         A    B    C    B    C    E
    0   A0   B0   C0  NaN  NaN  NaN
    1   A1   B1   C1  NaN  NaN  NaN
    2   A2   B2   C2   B2   C2   E2
    6  NaN  NaN  NaN   B6   C6   E6
    7  NaN  NaN  NaN   B7   C7   E7
    ==================================================
        A   B   C   B   C   E
    2  A2  B2  C2  B2  C2  E2
    ==================================================
         A    B    C    B    C    E
    0   A0   B0   C0  NaN  NaN  NaN
    1   A1   B1   C1  NaN  NaN  NaN
    2   A2   B2   C2   B2   C2   E2
    6  NaN  NaN  NaN   B6   C6   E6
    7  NaN  NaN  NaN   B7   C7   E7
    


```python
# 1. 데이터 가공

# 데이터 이어 붙이기(pd.concat()): 축의 로직 설정으로 이어 붙이기

# reindex: DataFrame의 인덱스를 재정렬하거나 변경할 때 사용
# df1의 index를 기준으로 결합
result = pd.concat([df1, df4], axis=1).reindex(df1.index)
print(result), print('='*50)

# ignore_index=True: 원본 인덱스를 무시하고 결합된 결과 DataFrame에
# 새로운 정수 인덱스(0부터 시작)를 자동으로 생성
# 인덱스가 중복되는 경우 무시할 수 있음
result2 = pd.concat([df1, df4], ignore_index=True)
print(result2)
```

        A   B   C    B    C    E
    0  A0  B0  C0  NaN  NaN  NaN
    1  A1  B1  C1  NaN  NaN  NaN
    2  A2  B2  C2   B2   C2   E2
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
# 공식적으로 제거(removed) 되었
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
result = pd.concat([df1, s2], axis=1)
print(result), print('='*30)

s3 = pd.Series(['*0', '*1', '*2'])
result2 = pd.concat([df1, s3, s3, s3], axis=1)
print(result2)
```

         A    B    C   Z
    0   A0   B0   C0  Z0
    1   A1   B1   C1  Z1
    2   A2   B2   C2  Z2
    3  NaN  NaN  NaN  Z3
    ==============================
        A   B   C   0   1   2
    0  A0  B0  C0  *0  *0  *0
    1  A1  B1  C1  *1  *1  *1
    2  A2  B2  C2  *2  *2  *2
    


```python
# 1. 데이터 가공

# 데이터 이어 붙이기(pd.concat()): 그룹 키로 이어 붙이기
s4 = pd.Series([0, 1, 2, 3], name='J')
s5 = pd.Series([0, 1, 2, 3])
s6 = pd.Series([0, 1, 4, 5])

result3 = pd.concat([s4, s5, s6], axis=1)
print(result3), print('='*30)

# keys = [ ]: DataFrame의 컬럼 이름을 명시적으로 지정
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

result = pd.concat(frames, keys=['ha', 'hi', 'ho'])
print(result), print('='*50)

# DataFrame들을 딕셔너리 형태로 준비
# 딕셔너리의 키('ha', 'hi', 'ho')는 해당 DataFrame의 상위 레벨 인덱스가 됨
pic = {'ha': df1, 'hi': df2, 'ho': df3}
result1 = pd.concat(pic)
print(result1), print('='*50)

# 딕셔너리의 키들 중 특정 키만 선택하여 포함하고, 그 순서도 재정렬
result2 = pd.concat(pic, keys=['ho', 'hi'])
print(result2)
```

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
