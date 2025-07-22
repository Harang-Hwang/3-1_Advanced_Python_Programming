```python
import numpy as np
import pandas as pd
```

### 1. 판다스 데이터 구조


```python
# Series

# 시리즈 객체 생성 방법: 1. ndarray에서 시리즈 객체 생성
s1 = pd.Series(np.random.randn(4), index=['a', 'b', 'c', 'd'])

print(s1), print('-'*50)

# 클래스 객체인 pandas.Series의 속성 values와 index를 이용해 각 값을 구할 수 있음
print(s1.values)
print(s1.index)

print(s1['b'])  # 라벨 'b'를 사용하여 값에 접근 (행 라벨로 접근)
print('='*50)

## 인덱스 미지정 (기본 정수 인덱스)
s2 = pd.Series([100, 200, 300])
print(s2)
print(s2.values)
print(s2.index)
print(s2[1])  # 기본 정수 인덱스 (행 라벨)를 사용하여 값에 접근
```

    a   -0.228686
    b    0.561790
    c    0.865264
    d    0.735201
    dtype: float64
    --------------------------------------------------
    [-0.22868575  0.56179042  0.86526394  0.73520063]
    Index(['a', 'b', 'c', 'd'], dtype='object')
    0.5617904163127239
    ==================================================
    0    100
    1    200
    2    300
    dtype: int64
    [100 200 300]
    RangeIndex(start=0, stop=3, step=1)
    200
    


```python
# Series

# 시리즈 객체 생성 방법: 2. dictionary 데이터에서 시리즈 객체 생성
dict_data = {'seoul': 2000, 'busan': 2500, 'daejeon': 3000}

s1 = pd.Series(dict_data)
print(s1), print('='*20)

# 인덱스 순서 지정
dict_data2 = {'a': 0., 'b': 1., 'c': 2.}
index_order = ['c', 'a', 'b']  # 원하는 인덱스 순서

series = pd.Series(dict_data2, index=index_order)
print(s2), print('='*20)

# 인덱스 'd'는 없는 인덱스 -> NaN (Not a Number, 표준 손실 값 표시자)로 표현됨
s3 = pd.Series(dict_data2, index=['b', 'c', 'd', 'a'])
print(s3)
print(s3.values)
print(s3.index)
```

    seoul      2000
    busan      2500
    daejeon    3000
    dtype: int64
    ====================
    a    0.0
    b    1.0
    c    2.0
    dtype: float64
    ====================
    b    1.0
    c    2.0
    d    NaN
    a    0.0
    dtype: float64
    [ 1.  2. nan  0.]
    Index(['b', 'c', 'd', 'a'], dtype='object')
    


```python
# Series

# 시리즈 객체 생성 방법: 3. 스칼라 값에서 시리즈 객체 생성

# 숫자 스칼라 값
scalar_value = 10
index_labels = ['a', 'b', 'c']
series = pd.Series(scalar_value, index=index_labels)
print(series), print('='*50)

# 문자열 스칼라 값
series2 = pd.Series('apple', index=[1, 2, 3, 4])
print(series2), print('='*50)

# 불리언 스칼라 값
scalar_value = True
index_labels = ['first', 'second']
series3 = pd.Series(scalar_value, index=index_labels)
print(series3)
```

    a    10
    b    10
    c    10
    dtype: int64
    ==================================================
    1    apple
    2    apple
    3    apple
    4    apple
    dtype: object
    first     True
    second    True
    dtype: bool
    


```python
# Series

# ndarray와의 유사성
s1 = pd.Series(np.random.randn(5), index=['a', 'b', 'c', 'd', 'e'])
print(s1), print('='*20)

print(f's1[0] : \n{s1[0]}')
print('='*20)

print(f's1[:3] : \n{s1[:3]}')
print('='*20)

print(f's1[s1 > s1.median()] : \n{s1[s1 > s1.median()]}')
print('='*20)

print(f's1[[4, 3, 1]] : \n{s1[[4, 3, 1]]}')
print('='*20)

print(f'np.exp(s1) : \n{np.exp(s1)}')
```

    a   -2.342135
    b    0.699945
    c   -0.092563
    d    0.368495
    e   -0.933173
    dtype: float64
    ====================
    s1[0] : 
    -2.3421350722674865
    ====================
    s1[:3] : 
    a   -2.342135
    b    0.699945
    c   -0.092563
    dtype: float64
    ====================
    s1[s1 > s1.median()] : 
    b    0.699945
    d    0.368495
    dtype: float64
    ====================
    s1[[4, 3, 1]] : 
    e   -0.933173
    d    0.368495
    b    0.699945
    dtype: float64
    ====================
    np.exp(s1) : 
    a    0.096122
    b    2.013643
    c    0.911591
    d    1.445558
    e    0.393304
    dtype: float64
    

    C:\Users\User\AppData\Local\Temp\ipykernel_23720\2646069006.py:7: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
      print(f's1[0] : \n{s1[0]}')
    C:\Users\User\AppData\Local\Temp\ipykernel_23720\2646069006.py:16: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
      print(f's1[[4, 3, 1]] : \n{s1[[4, 3, 1]]}')
    


```python
# Series

# dictionary와의 유사성

# 딕셔너리로 Series 객체 생성
data = {'a': 10, 'b': 20, 'c': 30}
series = pd.Series(data)
print(series), print("-"*30)

# 인덱스 라벨을 사용하여 값 얻기
value_b = series['b']
print(f"Value at index 'b': {value_b}"), print("-"*30)

# 인덱스 라벨을 사용하여 값 변경
series['b'] = 50
print(f"Series after changing value at 'b': \n{series}")
print("-"*30)

# 새로운 인덱스 라벨과 값 동적 할당 (크기 확장)
series['d'] = 40
print(f"Series after adding a new element: \n{series}")
```

    a    10
    b    20
    c    30
    dtype: int64
    ------------------------------
    Value at index 'b': 20
    ------------------------------
    Series after changing value at 'b': 
    a    10
    b    50
    c    30
    dtype: int64
    ------------------------------
    Series after adding a new element: 
    a    10
    b    50
    c    30
    d    40
    dtype: int64
    


```python
# Series

# 넘파이와의 유사성
s1 = pd.Series(np.random.randint(0,5,5), index=['a', 'b', 'c', 'd', 'e'])

# 주의: 인덱스의 순서가 s1과 조금 다름
s2 = pd.Series(np.random.randint(10,15,5), index=['a', 'd', 'c', 'e', 'b'])

print(f's1 : \n{s1}'), print('='*30)
print(f's2 : \n{s2}'), print('='*30)

print(f's1*2 : \n{s1*2}'), print('='*30)

# 시리즈를 연산하면 라벨에 기반해 데이터를 자동 정렬한다는 점에서 ndarray와 차이가 있음
print(f's1 + s2 : \n{s1 + s2}'), print('='*30)

# 시리즈들을 연산할 때 한 시리즈나 다른 시리즈에서 라벨이 발견되지 않으면 결과는 NaN으로 표시
print(s1[1:] + s1[:-1])
```

    s1 : 
    a    0
    b    2
    c    3
    d    4
    e    1
    dtype: int32
    ==============================
    s2 : 
    a    12
    d    14
    c    12
    e    10
    b    12
    dtype: int32
    ==============================
    s1*2 : 
    a    0
    b    4
    c    6
    d    8
    e    2
    dtype: int32
    ==============================
    s1 + s2 : 
    a    12
    b    14
    c    15
    d    18
    e    11
    dtype: int32
    ==============================
    a    NaN
    b    4.0
    c    6.0
    d    8.0
    e    NaN
    dtype: float64
    


```python
# Series

# 시리즈 이름 설정과 변경

# name 속성을 이용한 시리즈 이름 설정 및 확인
data = [10, 20, 30]
index = ['a', 'b', 'c']
s1 = pd.Series(data, index=index)
print(s1), print('-'*30)
print(f"Series name: {s1.name}")  # 초기에는 None
print('='*30)

# name 속성을 사용하여 이름 설정
s1.name = 'sample_data'
print(s1), print('-'*30)
print(f"Series name: {s1.name}")
print('='*30)

# rename() 메소드를 이용한 시리즈 이름 변경
s2 = s1.rename('new_data')
print(s2), print('-'*30)
print(f"New series name: {s2.name}")
print(f"Original series name: {s1.name}") # 원본 Series 이름은 그대로
```

    a    10
    b    20
    c    30
    dtype: int64
    ------------------------------
    Series name: None
    ==============================
    a    10
    b    20
    c    30
    Name: sample_data, dtype: int64
    ------------------------------
    Series name: sample_data
    ==============================
    a    10
    b    20
    c    30
    Name: new_data, dtype: int64
    ------------------------------
    New series name: new_data
    Original series name: sample_data
    


```python
# DataFrame

# 데이터프레임 객체 생성 방법:
# 1. 딕셔너리에서 데이터프레임 객체 생성

# a. 기본 형태: 값으로 리스트 배열 사용
data1 = {'이름': ['Wonyoung', 'Hayoung', 'Soyeon'],
        '나이': [22, 29, 26],
        '그룹': ['IVE', 'fromis_9', 'GIdle']}

df1 = pd.DataFrame(data1)
print(df1), print('='*50)

# b. 기본 형태: 값으로 NumPy 배열 사용
data2 = {'이름': np.array(['Wonyoung', 'Hayoung', 'Soyeon']),
        '나이': np.array([22, 29, 26]),
        '그룹': np.array(['IVE', 'fromis_9', 'GIdle'])}

df2 = pd.DataFrame(data2)
print(df2), print('='*50)

# c. 인덱스 명시하기
df3 = pd.DataFrame(data2, index = ['top1', 'top2', 'top3'])
print(df3)
```

             이름  나이        그룹
    0  Wonyoung  22       IVE
    1   Hayoung  29  fromis_9
    2    Soyeon  26     GIdle
    ==================================================
             이름  나이        그룹
    0  Wonyoung  22       IVE
    1   Hayoung  29  fromis_9
    2    Soyeon  26     GIdle
    ==================================================
                이름  나이        그룹
    top1  Wonyoung  22       IVE
    top2   Hayoung  29  fromis_9
    top3    Soyeon  26     GIdle
    


```python
# DataFrame

# 데이터프레임 객체 생성 방법:
# 1. 딕셔너리에서 데이터프레임 객체 생성

# d. 값으로 Series 객체의 딕셔너리 사용
# 딕셔너리에서 인덱스로 생성한 데이터프레임은 여러 시리즈 인덱스들의 합집합 (union)임
data3 = {'one': pd.Series([1, 2, 3], index=['a', 'b', 'c']),
        'two': pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])}

df4 = pd.DataFrame(data3)
print(df4), print('='*50)

print(df4.info()), print('-'*50)
print(df4.dtypes), print('-'*50)

# 인덱스와 열 속성을 이용해 각 행과 열 라벨에 접근할 수 있음
print(df4.index), print('='*50)
print(df4.columns)

# 인덱스와 열 라벨 순서를 변경할 수 있음
df5 = pd.DataFrame(data3, index = ['d', 'b', 'a'])
print(df5), print('='*50)

df6 = pd.DataFrame(data3, index = ['d', 'b', 'a'], columns = ['two', 'three'])
print(df6)
```

       one  two
    a  1.0    1
    b  2.0    2
    c  3.0    3
    d  NaN    4
    ==================================================
    <class 'pandas.core.frame.DataFrame'>
    Index: 4 entries, a to d
    Data columns (total 2 columns):
     #   Column  Non-Null Count  Dtype  
    ---  ------  --------------  -----  
     0   one     3 non-null      float64
     1   two     4 non-null      int64  
    dtypes: float64(1), int64(1)
    memory usage: 96.0+ bytes
    None
    --------------------------------------------------
    one    float64
    two      int64
    dtype: object
    --------------------------------------------------
    Index(['a', 'b', 'c', 'd'], dtype='object')
    ==================================================
    Index(['one', 'two'], dtype='object')
       one  two
    d  NaN    4
    b  2.0    2
    a  1.0    1
    ==================================================
       two three
    d    4   NaN
    b    2   NaN
    a    1   NaN
    


```python
# DataFrame

# df.dtypes vs. df.dtype

# df.dtypes
data = {'col1': [1, 2, 3],
        'col2': [1.1, 2.2, 3.3],
        'col3': ['a', 'b', 'c'],
        'col4': [True, False, True]}
df = pd.DataFrame(data)
print(df), print('-'*30)
print(df.dtypes), print('='*30)

# df.dtype
data2 = {'col1': [1, 2, 3]}
df_single_col = pd.DataFrame(data2)
# 해당 열을 Series로 선택한 후 dtype 확인
print(df_single_col['col1'].dtype)
```

       col1  col2 col3   col4
    0     1   1.1    a   True
    1     2   2.2    b  False
    2     3   3.3    c   True
    ------------------------------
    col1      int64
    col2    float64
    col3     object
    col4       bool
    dtype: object
    int64
    


```python
# DataFrame

# 데이터프레임 객체 생성 방법:
# 2. 구조화된 배열 또는 레코드 배열에서 데이터프레임 객체 생성

# 구조화된 배열의 데이터 타입 정의 및 구조화된 배열 생성
dtype = np.dtype([('성', 'U10'), ('나이', 'i4'), ('키', 'f8')])
data = np.array([('Jang',22,173.), ('Song',29,163.), ('Jeon',28,156.5)], dtype=dtype)

print(f"구조화된 NumPy 배열: {data}")
print(data.dtype), print('='*50)

# 구조화된 배열로부터 DataFrame 생성
df = pd.DataFrame(data)
print(f"생성된 DataFrame: \n{df}"), print('='*50)

# 인덱스 명시하기 (index 파라미터)
df2 = pd.DataFrame(data, index = ['first', 'second', 'third'])
print(f"DataFrame with index: \n{df2}"), print('='*50)

# 열 순서 지정
df3 = pd.DataFrame(data, columns = ['나이', '키', '성'])
print(f"DataFrame with changed columns: \n{df3}")
```

    구조화된 NumPy 배열: [('Jang', 22, 173. ) ('Song', 29, 163. ) ('Jeon', 28, 156.5)]
    [('성', '<U10'), ('나이', '<i4'), ('키', '<f8')]
    ==================================================
    생성된 DataFrame: 
          성  나이      키
    0  Jang  22  173.0
    1  Song  29  163.0
    2  Jeon  28  156.5
    ==================================================
    DataFrame with index: 
               성  나이      키
    first   Jang  22  173.0
    second  Song  29  163.0
    third   Jeon  28  156.5
    ==================================================
    DataFrame with changed columns: 
       나이      키     성
    0  22  173.0  Jang
    1  29  163.0  Song
    2  28  156.5  Jeon
    


```python
# DataFrame

# 데이터프레임 객체 생성 방법:
# 3. 딕셔너리를 요소로 갖는 리스트로 데이터프레임 객체 생성

data = [{'그룹': 'AESPA', '멤버수': 4, '데뷔연도': '2020'},
        {'그룹': 'KiiKii', '멤버수': 5, '데뷔연도': '2025'},
        {'그룹': 'IVE', '멤버수': 6, '데뷔연도': '2021'}]

df = pd.DataFrame(data)
print(df), print('='*50)

# 누락된 키 처리
data2 = [{'그룹': 'AESPA', '멤버수': 4},
         {'그룹': 'KiiKii', '데뷔연도': '2025'},
         {'그룹': 'IVE', '멤버수': 6, '데뷔연도': '2021'}]

df2 = pd.DataFrame(data2)
print(df2)


```

           그룹  멤버수  데뷔연도
    0   AESPA    4  2020
    1  KiiKii    5  2025
    2     IVE    6  2021
    ==================================================
           그룹  멤버수  데뷔연도
    0   AESPA  4.0   NaN
    1  KiiKii  NaN  2025
    2     IVE  6.0  2021
    


```python
# DataFrame

# 데이터프레임 객체 생성 방법:
# 4. 튜플의 딕셔너리에서 데이터프레임 객체 생성

# 튜플을 DataFrame의 열 이름으로 사용하는 경우
data = {('a', 'x'): [1, 2, 3],
        ('a', 'y'): [4, 5, 6],
        ('b', 'z'): [7, 8, 9]}

df = pd.DataFrame(data)
print(df), print('='*50)


df2 = pd.DataFrame({('a', 'b'): {('A', 'B'): 1, ('A', 'C'): 2},
                    ('a', 'a'): {('A', 'C'): 3, ('A', 'B'): 4},
                    ('a', 'c'): {('A', 'B'): 5, ('A', 'C'): 6},
                    ('b', 'a'): {('A', 'C'): 7, ('A', 'B'): 8},
                    ('b', 'b'): {('A', 'D'): 9, ('A', 'B'): 10}})

print(df2), print('-'*50)
print(df2.index), print('-'*50)
print(df2.columns)
```

       a     b
       x  y  z
    0  1  4  7
    1  2  5  8
    2  3  6  9
    ==================================================
           a              b      
           b    a    c    a     b
    A B  1.0  4.0  5.0  8.0  10.0
      C  2.0  3.0  6.0  7.0   NaN
      D  NaN  NaN  NaN  NaN   9.0
    --------------------------------------------------
    MultiIndex([('A', 'B'),
                ('A', 'C'),
                ('A', 'D')],
               )
    --------------------------------------------------
    MultiIndex([('a', 'b'),
                ('a', 'a'),
                ('a', 'c'),
                ('b', 'a'),
                ('b', 'b')],
               )
    


```python
# DataFrame

# 데이터프레임 객체 생성 방법:
# 5-1. 데이터프레임 생성자로부터 객체 생성 (DataFrame.from_dict 생성자)
data1 = {'이름': ['Wonyoung', 'Hayoung', 'Soyeon'],
        '나이': [22, 29, 26],
        '그룹': ['IVE', 'fromis_9', 'GIdle']}

df1 = pd.DataFrame.from_dict(data1)
print(df1), print('='*50)

# orient='index': 딕셔너리의 키를 행 인덱스(라벨)로, 값을 행 데이터로 해석
df2 = pd.DataFrame.from_dict(data1, orient='index')
print(df2), print('='*50)

# columns에는 원하는 열 이름을 리스트로 지정
df3 = pd.DataFrame.from_dict(data1, orient='index', columns=['one','two','three'])
print(df3), print('='*50)



# 5-2. 데이터프레임 생성자로부터 객체 생성 (DataFrame.from_records 생성자)
dtype = np.dtype([('성', 'U10'), ('나이', 'i4'), ('키', 'f8')])
data4 = np.array([('Jang',22,173.), ('Song',29,163.), ('Jeon',28,156.5)], dtype=dtype)

df4 = pd.DataFrame.from_records(data4)
print(df4)

df5 = pd.DataFrame.from_records(data4, index = ['top1','top2','top3'])
print(df5), print('='*50)

df6 = pd.DataFrame.from_records(data4, index = '성')
print(df6)
```

             이름  나이        그룹
    0  Wonyoung  22       IVE
    1   Hayoung  29  fromis_9
    2    Soyeon  26     GIdle
    ==================================================
               0         1       2
    이름  Wonyoung   Hayoung  Soyeon
    나이        22        29      26
    그룹       IVE  fromis_9   GIdle
    ==================================================
             one       two   three
    이름  Wonyoung   Hayoung  Soyeon
    나이        22        29      26
    그룹       IVE  fromis_9   GIdle
    ==================================================
          성  나이      키
    0  Jang  22  173.0
    1  Song  29  163.0
    2  Jeon  28  156.5
             성  나이      키
    top1  Jang  22  173.0
    top2  Song  29  163.0
    top3  Jeon  28  156.5
    ==================================================
          나이      키
    성              
    Jang  22  173.0
    Song  29  163.0
    Jeon  28  156.5
    


```python
# 행과 열의 기본 처리

# 행 또는 열 선택, 추가
d = {'one': pd.Series([1., 2., 3.], index = ['a', 'b', 'c']),
     'two': pd.Series([1., 2., 3., 4.], index = ['a', 'b', 'c', 'd'])}

df = pd.DataFrame(d)
print(df), print('='*50)

# 단일 열 선택
print(df['one']), print('='*50)

# 여러 열 선택 (대괄호 주의)
print(df[['one','two']]), print('='*50)

df['three'] = df['one'] * df['two']
print(df), print('='*50)

df['flag'] = df['one'] > 2
print(df), print('='*50)


# 행 또는 열 삭제 (del)
del df['two']
print(df), print('='*50)

# pandas.DataFrame.pop 메소드는 열을 추출하고 그 요소를 시리즈로 반환
three = df.pop('three')

print(three)
print(three.values)
print(type(three))

# 데이터프레임에 스칼라 값을 동적할당하면 브로드캐스팅으로 열을 채움
df['ha'] = 'hiho'

print(df)
```

       one  two
    a  1.0  1.0
    b  2.0  2.0
    c  3.0  3.0
    d  NaN  4.0
    ==================================================
    a    1.0
    b    2.0
    c    3.0
    d    NaN
    Name: one, dtype: float64
    ==================================================
       one  two
    a  1.0  1.0
    b  2.0  2.0
    c  3.0  3.0
    d  NaN  4.0
    ==================================================
       one  two  three
    a  1.0  1.0    1.0
    b  2.0  2.0    4.0
    c  3.0  3.0    9.0
    d  NaN  4.0    NaN
    ==================================================
       one  two  three   flag
    a  1.0  1.0    1.0  False
    b  2.0  2.0    4.0  False
    c  3.0  3.0    9.0   True
    d  NaN  4.0    NaN  False
    ==================================================
       one  three   flag
    a  1.0    1.0  False
    b  2.0    4.0  False
    c  3.0    9.0   True
    d  NaN    NaN  False
    ==================================================
    a    1.0
    b    4.0
    c    9.0
    d    NaN
    Name: three, dtype: float64
    [ 1.  4.  9. nan]
    <class 'pandas.core.series.Series'>
       one   flag    ha
    a  1.0  False  hiho
    b  2.0  False  hiho
    c  3.0   True  hiho
    d  NaN  False  hiho
    


```python
# 행과 열의 기본 처리

# 데이터프레임과 다른 인덱스를 가진 시리즈를 삽입할 때는 데이터프레임의 인덱스에 맞춤
print(df), print('='*50)

df['truncated_one'] = df['one'][:2]   # df['one'][:2]는 시리즈 객체 타입임
print(df)
```

       one   flag    ha
    a  1.0  False  hiho
    b  2.0  False  hiho
    c  3.0   True  hiho
    d  NaN  False  hiho
    ==================================================
       one   flag    ha  truncated_one
    a  1.0  False  hiho            1.0
    b  2.0  False  hiho            2.0
    c  3.0   True  hiho            NaN
    d  NaN  False  hiho            NaN
    


```python
# 행과 열의 기본 처리

# DataFrame.insert 함수: 특정 위치에 열을 삽입
print(df), print('='*50)

# 1: 1번째 열, 'hi': 삽입할 열 라벨, df['one']: 삽입할 값(시리즈 형)
df.insert(1, 'hi', df['one'])
print(df)
```

       one   flag    ha  truncated_one
    a  1.0  False  hiho            1.0
    b  2.0  False  hiho            2.0
    c  3.0   True  hiho            NaN
    d  NaN  False  hiho            NaN
    ==================================================
       one   hi   flag    ha  truncated_one
    a  1.0  1.0  False  hiho            1.0
    b  2.0  2.0  False  hiho            2.0
    c  3.0  3.0   True  hiho            NaN
    d  NaN  NaN  False  hiho            NaN
    


```python
# 행과 열의 기본 처리

# pandas.Series.drop 함수:
# 인덱스 라벨을 기준으로 시리즈의 요소를 제거하며, 요소를 제거한 시리즈 객체를 결과로 반환
ser = pd.Series(data = np.arange(3), index = ['A', 'B', 'C'])
print(ser), print('='*50)

ser2 = ser.drop(['B', 'C'])
print(ser2), print('='*50)

# pandas.DataFrame.drop:
# 라벨 이름과 축을 입력하거나 직접 인덱스나 열 이름을 입력해 행이나 열을 제거
df1 = pd.DataFrame(np.arange(12).reshape(3, 4), columns = ['A', 'B', 'C', 'D'])
print(df1), print('='*50)

df2 = df1.drop(['B', 'C'], axis = 1)  # 라벨 이름과 축으로 열 제거
print(df2), print('='*50)

df3 = df1.drop([0, 1])  # 인덱스로 행 제거
print(df3)
```

    A    0
    B    1
    C    2
    dtype: int64
    ==================================================
    A    0
    dtype: int64
    ==================================================
       A  B   C   D
    0  0  1   2   3
    1  4  5   6   7
    2  8  9  10  11
    ==================================================
       A   D
    0  0   3
    1  4   7
    2  8  11
    ==================================================
       A  B   C   D
    2  8  9  10  11
    
