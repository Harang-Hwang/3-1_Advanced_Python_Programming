### 10주차 강의 Pandas 2


```python
import numpy as np
import pandas as pd
```


```python
# DataFrame

# 데이터프레임 객체 생성 방법:
# 1. 딕셔너리에서 데이터프레임 객체 생성

# a. 기본 형태: 값으로 리스트 배열 사용
data1 = {'이름': ['Wonyoung', 'Hayoung', 'Soyeon'],
        '나이': [22, 29, 26],
        '그룹': ['IVE', 'fromis_9', 'GIdle']}

df1 = pd.DataFrame(data1)
print(f'값으로 리스트 배열 사용: \n{df1}'), print('='*50)

# b. 기본 형태: 값으로 NumPy 배열 사용
data2 = {'이름': np.array(['Wonyoung', 'Hayoung', 'Soyeon']),
        '나이': np.array([22, 29, 26]),
        '그룹': np.array(['IVE', 'fromis_9', 'GIdle'])}

df2 = pd.DataFrame(data2)
print(f'값으로 NumPy 배열 사용: \n{df2}'), print('='*50)

# c. 인덱스 명시하기
df3 = pd.DataFrame(data2, index = ['top1', 'top2', 'top3'])
print(f'인덱스 명시하기: \n{df3}')
```

    값으로 리스트 배열 사용: 
             이름  나이        그룹
    0  Wonyoung  22       IVE
    1   Hayoung  29  fromis_9
    2    Soyeon  26     GIdle
    ==================================================
    값으로 NumPy 배열 사용: 
             이름  나이        그룹
    0  Wonyoung  22       IVE
    1   Hayoung  29  fromis_9
    2    Soyeon  26     GIdle
    ==================================================
    인덱스 명시하기: 
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
print(f'값으로 Series 객체 사용: \n{df4}'), print('-'*50)

print(df4.info()), print('-'*50)
print(df4.dtypes), print('='*50)

# 인덱스와 열 속성을 이용해 각 행과 열 라벨에 접근할 수 있음
print(df4.index), print('-'*50)
print(df4.columns), print('='*50)

# 인덱스와 열 라벨 순서를 변경할 수 있음
df5 = pd.DataFrame(data3, index = ['d', 'b', 'a'])
print(df5), print('='*50)

df6 = pd.DataFrame(data3, index = ['d', 'b', 'a'], columns = ['two', 'three'])
print(df6)
```

    값으로 Series 객체 사용: 
       one  two
    a  1.0    1
    b  2.0    2
    c  3.0    3
    d  NaN    4
    --------------------------------------------------
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
    ==================================================
    Index(['a', 'b', 'c', 'd'], dtype='object')
    --------------------------------------------------
    Index(['one', 'two'], dtype='object')
    ==================================================
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
print(df.dtypes), print('-'*30)
print(f'df.dtypes 자료형: \n{type(df.dtypes)}'),print('='*30)

# df.dtype
# DataFrame 객체에서 하나의 열을 선택한 후 dtype 확인
print(df['col1'].dtype)
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
    ------------------------------
    df.dtypes 자료형: 
    <class 'pandas.core.series.Series'>
    ==============================
    int64
    


```python
# DataFrame

# 데이터프레임 객체 생성 방법:
# 2. 구조화된 배열 또는 레코드 배열에서 데이터프레임 객체 생성

# 구조화된 배열의 데이터 타입 정의 및 구조화된 배열 생성
dtype = np.dtype([('성', 'U10'), ('나이', 'i4'), ('키', 'f8')])
data = np.array([('Jang', 22, 173.), ('Song', 29, 163.), ('Jeon', 28, 156.5)],
                dtype=dtype)

print(f"구조화된 NumPy 배열: \n{data}"), print('-'*50)
print(data.dtype), print('='*50)

# 구조화된 배열로부터 DataFrame 생성
df = pd.DataFrame(data)
print(f"DataFrame: \n{df}"), print('='*50)

# 인덱스 명시하기 (index 파라미터)
df2 = pd.DataFrame(data, index = ['first', 'second', 'third'])
print(f"DataFrame with index: \n{df2}"), print('='*50)

# 열 순서 지정
df3 = pd.DataFrame(data, columns = ['나이', '키', '성'])
print(f"DataFrame with changed columns: \n{df3}")
```

    구조화된 NumPy 배열: 
    [('Jang', 22, 173. ) ('Song', 29, 163. ) ('Jeon', 28, 156.5)]
    --------------------------------------------------
    [('성', '<U10'), ('나이', '<i4'), ('키', '<f8')]
    ==================================================
    DataFrame: 
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
data2 = [{'그룹': 'AESPA', '멤버수': 4},            # 데뷔연도 누락됨
         {'그룹': 'KiiKii', '데뷔연도': '2025'},    # 멤버수 누락됨
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
# 4. 튜플을 키로 갖는 딕셔너리에서 데이터프레임 객체 생성

# 튜플을 DataFrame의 열 이름으로 사용하는 경우
data = {('a'): [1, 2, 3],
        ('b'): [4, 5, 6],
        ('c'): [7, 8, 9]}
df = pd.DataFrame(data)
print(df)
```

       a  b  c
    0  1  4  7
    1  2  5  8
    2  3  6  9
    


```python
# DataFrame

# 데이터프레임 객체 생성 방법:
# 4. 튜플을 키로 갖는 딕셔너리에서 데이터프레임 객체 생성

# MultiIndex를 사용하는 경우
# 튜플을 열 이름으로 사용하는 경우
data_columns = {('IVE', 'Liz'): ['Jeju', 22],   # 각 열이 동일한 데이터 타입이 아님
                ('IVE', 'Rei'): ['Nagoya', 23],
                ('AESPA', 'Winter'): ['Busan', 26]}

df_columns = pd.DataFrame(data_columns)
print(f"MultiIndex DataFrame: \n{df_columns}"), print('-'*50)
print(df_columns.dtypes), print('='*50)

###### 참고) 위에서.. ###########################################
# 하나의 열에 여러 타입의 데이터가 섞여 있음
# Pandas는 그 모든 데이터를 담을 수 있는 가장 일반적인 타입
# (주로 object)으로 자동 변환(업캐스팅)하여 DataFrame을 생성
# ojbect는 모든 파이썬 클래스의 최상위 부모 클래스임
#################################################################


# 튜플을 다중 인덱스의 레벨로 사용하는 경우 (Series의 딕셔너리 형태)
data_index = {'Birthplace': pd.Series(['Jeju', 'Nagoya'],
                                      index=[('IVE', 'Liz'), ('IVE', 'Rei')]),
              'Age': pd.Series([22, 26], index=[('IVE', 'Liz'), ('AESPA', 'Winter')])}
df_index = pd.DataFrame(data_index)
print("튜플을 다중 인덱스의 레벨로 사용한 DataFrame:")
print(df_index), print('-'*50)
print(df_index.dtypes)
```

    MultiIndex DataFrame: 
        IVE          AESPA
        Liz     Rei Winter
    0  Jeju  Nagoya  Busan
    1    22      23     26
    --------------------------------------------------
    IVE    Liz       object
           Rei       object
    AESPA  Winter    object
    dtype: object
    ==================================================
    튜플을 다중 인덱스의 레벨로 사용한 DataFrame:
                    Birthplace   Age
    (AESPA, Winter)        NaN  26.0
    (IVE, Liz)            Jeju  22.0
    (IVE, Rei)          Nagoya   NaN
    --------------------------------------------------
    Birthplace     object
    Age           float64
    dtype: object
    


```python
# DataFrame

# 데이터프레임 객체 생성 방법:
# 4. 튜플을 키로 갖는 딕셔너리에서 데이터프레임 객체 생성

# MultiIndex를 사용하는 경우 (교재의 예시)
data2 = {('a', 'x'): [1, 2, 3],
        ('a', 'y'): [4, 5, 6],
        ('b', 'z'): [7, 8, 9]}

df2 = pd.DataFrame(data2)
print(df2), print('='*50)

df3 = pd.DataFrame({('a', 'b'): {('A', 'B'): 1, ('A', 'C'): 2},
                    ('a', 'a'): {('A', 'C'): 3, ('A', 'B'): 4},
                    ('a', 'c'): {('A', 'B'): 5, ('A', 'C'): 6},
                    ('b', 'a'): {('A', 'C'): 7, ('A', 'B'): 8},
                    ('b', 'b'): {('A', 'D'): 9, ('A', 'B'): 10}})

print(df3), print('-'*50)
print(df3.index), print('-'*50)
print(df3.columns)
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

df0 = pd.DataFrame.from_dict(data1)
df1 = pd.DataFrame.from_dict(data1, orient='columns')   # orinet='columns' 기본값
print(df0), print('-'*50)
print(df1), print('='*50)

# orient='index': 딕셔너리의 키를 행 인덱스(라벨)로, 값을 행 데이터로 해석
df2 = pd.DataFrame.from_dict(data1, orient='index')
print(df2), print('='*50)

# orient='index'에서 columns에는 원하는 열 이름을 리스트로 지정
df3 = pd.DataFrame.from_dict(data1, orient='index', columns=['one','two','three'])
print(df3)
```

             이름  나이        그룹
    0  Wonyoung  22       IVE
    1   Hayoung  29  fromis_9
    2    Soyeon  26     GIdle
    --------------------------------------------------
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
    


```python
# DataFrame

# 데이터프레임 객체 생성 방법:
# 5-2. 데이터프레임 생성자로부터 객체 생성 (DataFrame.from_records 생성자)
dtype = np.dtype([('성', 'U10'), ('나이', 'i4'), ('키', 'f8')])
data4 = np.array([('Jang', 22, 173.), ('Song', 29, 163.), ('Jeon', 28, 156.5)],
                 dtype=dtype)
print(data4), print('='*30)

df4 = pd.DataFrame.from_records(data4)
print(df4), print('='*30)

df5 = pd.DataFrame.from_records(data4, index = ['top1','top2','top3'])
print(df5), print('='*30)

df6 = pd.DataFrame.from_records(data4, index = '성')
print(df6)
```

    [('Jang', 22, 173. ) ('Song', 29, 163. ) ('Jeon', 28, 156.5)]
    ==============================
          성  나이      키
    0  Jang  22  173.0
    1  Song  29  163.0
    2  Jeon  28  156.5
    ==============================
             성  나이      키
    top1  Jang  22  173.0
    top2  Song  29  163.0
    top3  Jeon  28  156.5
    ==============================
          나이      키
    성              
    Jang  22  173.0
    Song  29  163.0
    Jeon  28  156.5
    


```python
# 행과 열의 기본 처리

# 열 선택
d = {'one': pd.Series([1., 2., 3.], index = ['a', 'b', 'c']),
     'two': pd.Series([1., 2., 3., 4.], index = ['a', 'b', 'c', 'd'])}

df = pd.DataFrame(d)
print(df), print('='*30)

# 단일 열 선택
print(df['one']), print('='*30)

# 여러 열 선택 (대괄호 주의): 열 이름들을 리스트 형태로 묶음
print(df[['one','two']]), print('='*30)
#print(df['one','two']), print('='*30)
```

       one  two
    a  1.0  1.0
    b  2.0  2.0
    c  3.0  3.0
    d  NaN  4.0
    ==============================
    a    1.0
    b    2.0
    c    3.0
    d    NaN
    Name: one, dtype: float64
    ==============================
       one  two
    a  1.0  1.0
    b  2.0  2.0
    c  3.0  3.0
    d  NaN  4.0
    ==============================
       one  two  three
    a  1.0  1.0    1.0
    b  2.0  2.0    4.0
    c  3.0  3.0    9.0
    d  NaN  4.0    NaN
    ==============================
       one  two  three   flag
    a  1.0  1.0    1.0  False
    b  2.0  2.0    4.0  False
    c  3.0  3.0    9.0   True
    d  NaN  4.0    NaN  False
    ========================================
       one  two  three   flag  truncated_one
    a  1.0  1.0    1.0  False            1.0
    b  2.0  2.0    4.0  False            2.0
    c  3.0  3.0    9.0   True            NaN
    d  NaN  4.0    NaN  False            NaN
    


```python
# 행과 열의 기본 처리

# 행 선택

# DataFrame.loc[] 속성 vs. DataFrame.iloc[] 속성
data = {'A': [10, 20, 30],
        'B': [100, 200, 300]}
index=['x', 'y', 'z']

df = pd.DataFrame(data, index = index)

# DataFrame.loc[] 속성
print(df), print('-'*30)
print(df.loc['x']), print('-'*30)       # index 라벨이 'x'인 행 반환
print(df.loc['x':'y']), print('-'*30)   # 'x'부터 'y'까지 ('y'까지 포함)
print(df.loc['x', 'B']), print('='*30)  # 'x'행, 'B'열 값 선택 -> 100

# DataFrame.iloc[] 속성
print(df.iloc[0]), print('-'*30)        # 0번째 행 반환
print(df.iloc[0:2]), print('-'*30)      # 0~1번째 행 (2번째는 제외)
print(df.iloc[0, 1])                    # 0행 1열 -> 100
```

        A    B
    x  10  100
    y  20  200
    z  30  300
    ------------------------------
    A     10
    B    100
    Name: x, dtype: int64
    ------------------------------
        A    B
    x  10  100
    y  20  200
    ------------------------------
    100
    ==============================
    A     10
    B    100
    Name: x, dtype: int64
    ------------------------------
        A    B
    x  10  100
    y  20  200
    ------------------------------
    100
    


```python
# 행과 열의 기본 처리

# 행 선택

# DataFrame.loc[] 속성 vs. DataFrame.iloc[] 속성
df = pd.DataFrame({'name': ['A', 'B', 'C', 'D'],
                    'age': [23, 35, 19, 42]})
print(df), print('='*30)

# DataFrame.loc[] 속성 : 라벨 기반이지만, 조건이 들어가면
#               Boolean Series를 받아 해당하는 행만 선택
print(df.loc[df['age'] >= 30])  # age가 30 이상인 행만 선택
print('='*30)

# DataFrame.iloc[] 속성 : Boolean 배열을 사용할 수 있지만,
#                        정수 위치 기준의 Boolean 배열이어야 함
mask = [False, True, False, True] # df에서 1번과 3번 행만 True로 설정
print(df.iloc[mask])
```

      name  age
    0    A   23
    1    B   35
    2    C   19
    3    D   42
    ==============================
      name  age
    1    B   35
    3    D   42
    ==============================
      name  age
    1    B   35
    3    D   42
    


```python
# 행과 열의 기본 처리

# 행 또는 열 추가
# 열 추가
d = {'one': pd.Series([1., 2., 3.], index = ['a', 'b', 'c']),
     'two': pd.Series([1., 2., 3., 4.], index = ['a', 'b', 'c', 'd'])}

df = pd.DataFrame(d)
print(df), print('='*30)

df['three'] = df['one'] * df['two']
print(df), print('='*30)

# 열 추가
df['flag'] = df['one'] > 2
print(df), print('='*40)

# 데이터프레임과 다른 인덱스를 가진 시리즈를 삽입할 때는 데이터프레임의 인덱스에 맞춤
df['truncated_one'] = df['one'][:2]   # df['one'][:2]는 시리즈 객체 타입임
print(df)

# 데이터프레임에 스칼라 값을 동적할당하면 브로드캐스팅으로 열을 채움
df['ha'] = 'hiho'
print(df)
```

       one  two
    a  1.0  1.0
    b  2.0  2.0
    c  3.0  3.0
    d  NaN  4.0
    ==============================
       one  two  three
    a  1.0  1.0    1.0
    b  2.0  2.0    4.0
    c  3.0  3.0    9.0
    d  NaN  4.0    NaN
    ==============================
       one  two  three   flag
    a  1.0  1.0    1.0  False
    b  2.0  2.0    4.0  False
    c  3.0  3.0    9.0   True
    d  NaN  4.0    NaN  False
    ========================================
       one  two  three   flag  truncated_one
    a  1.0  1.0    1.0  False            1.0
    b  2.0  2.0    4.0  False            2.0
    c  3.0  3.0    9.0   True            NaN
    d  NaN  4.0    NaN  False            NaN
       one  two  three   flag  truncated_one    ha
    a  1.0  1.0    1.0  False            1.0  hiho
    b  2.0  2.0    4.0  False            2.0  hiho
    c  3.0  3.0    9.0   True            NaN  hiho
    d  NaN  4.0    NaN  False            NaN  hiho
    


```python
# 행과 열의 기본 처리

# DataFrame.insert() 메소드: 특정 위치에 열을 삽입
print(df), print('='*50)

# 1: 1번째 열
# 'hi': 삽입할 열 라벨
# df['one']: 삽입할 값(시리즈 형)
df.insert(1, 'hi', df['one'])
print(df)
```

       one  two  three   flag  truncated_one    ha
    a  1.0  1.0    1.0  False            1.0  hiho
    b  2.0  2.0    4.0  False            2.0  hiho
    c  3.0  3.0    9.0   True            NaN  hiho
    d  NaN  4.0    NaN  False            NaN  hiho
    ==================================================
       one   hi  two  three   flag  truncated_one    ha
    a  1.0  1.0  1.0    1.0  False            1.0  hiho
    b  2.0  2.0  2.0    4.0  False            2.0  hiho
    c  3.0  3.0  3.0    9.0   True            NaN  hiho
    d  NaN  NaN  4.0    NaN  False            NaN  hiho
    


```python
# 행과 열의 기본 처리

# 행 또는 열 추가
# 행 추가
d = {'one': pd.Series([1., 2., 3.], index=['a', 'b', 'c']),
     'two': pd.Series([1., 2., 3., 4.], index=['a', 'b', 'c', 'd'])}

df = pd.DataFrame(d)

df['three'] = df['one'] * df['two']
print(f"원본 DataFrame: \n{df}"), print('='*30)

# 새로운 행 'e' 추가 (모든 기존 열에 값 지정)
df.loc['e'] = [4.0, 5.0, 6.0]
print(f"e 행 추가: \n{df}"), print('='*30)

# 새로운 행 'f' 추가 (기존 열 중 일부에만 값 지정, 나머지는 NaN)
df.loc['f'] = {'one': 5.0, 'two': 6.0}
print(f"f 행 추가: \n{df}"), print('='*30)

# iloc[]는 에러 발생
df.iloc[6] = {'one': 5.0, 'two': 6.0}
print(f"g 행 추가: \n{df}"), print('='*30)
```

    원본 DataFrame: 
       one  two  three
    a  1.0  1.0    1.0
    b  2.0  2.0    4.0
    c  3.0  3.0    9.0
    d  NaN  4.0    NaN
    ==============================
    e 행 추가: 
       one  two  three
    a  1.0  1.0    1.0
    b  2.0  2.0    4.0
    c  3.0  3.0    9.0
    d  NaN  4.0    NaN
    e  4.0  5.0    6.0
    ==============================
    f 행 추가: 
       one  two  three
    a  1.0  1.0    1.0
    b  2.0  2.0    4.0
    c  3.0  3.0    9.0
    d  NaN  4.0    NaN
    e  4.0  5.0    6.0
    f  5.0  6.0    NaN
    ==============================
    


    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    <ipython-input-26-6ebf6c397ace> in <cell line: 0>()
         19 print(f"f 행 추가: \n{df}"), print('='*30)
         20 
    ---> 21 df.iloc[6] = {'one': 5.0, 'two': 6.0}
         22 print(f"g 행 추가: \n{df}"), print('='*30)
    

    /usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in __setitem__(self, key, value)
        906             key = self._check_deprecated_callable_usage(key, maybe_callable)
        907         indexer = self._get_setitem_indexer(key)
    --> 908         self._has_valid_setitem_indexer(key)
        909 
        910         iloc = self if self.name == "iloc" else self.obj.iloc
    

    /usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _has_valid_setitem_indexer(self, indexer)
       1644             elif is_integer(i):
       1645                 if i >= len(ax):
    -> 1646                     raise IndexError("iloc cannot enlarge its target object")
       1647             elif isinstance(i, dict):
       1648                 raise IndexError("iloc cannot enlarge its target object")
    

    IndexError: iloc cannot enlarge its target object



```python
# 행과 열의 기본 처리

# del 키워드
del df['two']
print(df), print('='*50)

# pandas.DataFrame.pop 메소드: 열을 추출하고 그 요소를 시리즈로 반환
three = df.pop('three')

print(three), print('-'*50)
print(three.values), print('-'*50)
print(type(three)), print('='*50)
```

       one  three
    a  1.0    1.0
    b  2.0    4.0
    c  3.0    9.0
    d  NaN    NaN
    e  4.0    6.0
    f  5.0    NaN
    ==================================================
    a    1.0
    b    4.0
    c    9.0
    d    NaN
    e    6.0
    f    NaN
    Name: three, dtype: float64
    --------------------------------------------------
    [ 1.  4.  9. nan  6. nan]
    --------------------------------------------------
    <class 'pandas.core.series.Series'>
    ==================================================
    




    (None, None)




```python
# 행과 열의 기본 처리

# pandas.Series.drop 함수:
# 인덱스 라벨을 기준으로 시리즈의 요소를 제거하며, 요소를 제거한 시리즈 객체를 결과로 반환
s1 = pd.Series(data = np.arange(3), index = ['A', 'B', 'C'])
print(s1), print('-'*30)

s2 = s1.drop(['B', 'C'])
print(s2), print('='*30)

# pandas.DataFrame.drop:
# 라벨 이름과 축을 입력하거나 직접 인덱스나 열 이름을 입력해 행이나 열을 제거
df1 = pd.DataFrame(np.arange(12).reshape(3, 4), columns = ['A', 'B', 'C', 'D'])
print(df1), print('-'*30)

df2 = df1.drop(['B', 'C'], axis = 1)  # 라벨 이름과 축으로 열 제거
print(df2), print('='*30)

df3 = df1.drop([0, 1])  # 인덱스로 행 제거 (axis=0이 기본값)
print(df3)
```

    A    0
    B    1
    C    2
    dtype: int64
    ------------------------------
    A    0
    dtype: int64
    ==============================
       A  B   C   D
    0  0  1   2   3
    1  4  5   6   7
    2  8  9  10  11
    ------------------------------
       A   D
    0  0   3
    1  4   7
    2  8  11
    ==============================
       A  B   C   D
    2  8  9  10  11
    


```python
# 행과 열의 기본 처리

# 데이터 정렬 및 산술 연산
df = pd.DataFrame(np.random.randn(5, 4), columns = ['A', 'B', 'C', 'D'])

df2 = pd.DataFrame(np.random.randn(3, 3), columns = ['A', 'B', 'C'])

print(df), print('='*50)
print(df2), print('='*50)

# 데이터프레임 객체 간 연산을 실행하면, 행과 열 기준으로 자동 정렬됨
print(df + df2), print('='*50)


# 데이트프레임과 시리즈 간 연산을 실행하면, 데이터프레임 열에 시리즈 인덱스를 정렬
print(df - df.iloc[0]) # 행 방향으로 브로드캐스팅하는 것과 같음


# 데이트프레임과 스칼라 간 연산
print(df*10 + 2)
```

              A         B         C         D
    0  0.884647  1.854421 -1.516236 -0.414904
    1  0.322282 -0.059016 -0.408056 -1.354552
    2 -0.530595 -0.590024 -0.967021  0.044444
    3 -0.083054  0.828482 -0.047571 -0.603019
    4 -0.931594  0.159271  1.527698  1.665925
    ==================================================
              A         B         C
    0  0.909141  0.698673  0.763569
    1 -0.476303 -2.137070 -1.113817
    2  0.493049 -1.104463 -1.802996
    ==================================================
              A         B         C   D
    0  1.793787  2.553095 -0.752667 NaN
    1 -0.154021 -2.196085 -1.521873 NaN
    2 -0.037546 -1.694487 -2.770017 NaN
    3       NaN       NaN       NaN NaN
    4       NaN       NaN       NaN NaN
    ==================================================
              A         B         C         D
    0  0.000000  0.000000  0.000000  0.000000
    1 -0.562365 -1.913437  1.108180 -0.939648
    2 -1.415242 -2.444446  0.549215  0.459348
    3 -0.967701 -1.025939  1.468665 -0.188116
    4 -1.816241 -1.695150  3.043935  2.080829
               A          B          C          D
    0  10.846465  20.544213 -13.162361  -2.149036
    1   5.222818   1.409842  -2.080561 -11.545518
    2  -3.305955  -3.900242  -7.670207   2.444443
    3   1.169460  10.284824   1.524291  -4.030194
    4  -7.315942   3.592708  17.276985  18.659249
    


```python
# 행과 열의 기본 처리

# 데이터 정렬 및 산술 연산 (불리언 연산자)
df1 = pd.DataFrame({'a': [1, 0, 1], 'b': [0, 1, 1]}, dtype = bool)
df2 = pd.DataFrame({'a': [0, 1, 1], 'b': [1, 1, 0]}, dtype = bool)

print(df1), print('='*50)
print(df2), print('='*50)

print(df1 & df2), print('='*50)
print(df1 | df2), print('='*50)
print(df1 ^ df2), print('='*50)
print(-df1), print('='*50)
print(~df1)
```

           a      b
    0   True  False
    1  False   True
    2   True   True
    ==================================================
           a      b
    0  False   True
    1   True   True
    2   True  False
    ==================================================
           a      b
    0  False  False
    1  False   True
    2   True  False
    ==================================================
          a     b
    0  True  True
    1  True  True
    2  True  True
    ==================================================
           a      b
    0   True   True
    1   True  False
    2  False   True
    ==================================================
           a      b
    0  False   True
    1   True  False
    2  False  False
    ==================================================
           a      b
    0  False   True
    1   True  False
    2  False  False
    


```python
# 행과 열의 기본 처리

# 데이터 정렬 및 산술 연산 (전치)
df = pd.DataFrame(np.arange(12).reshape(3,4), columns = ['A', 'B', 'C', 'D'])

print(df), print('='*50)
print(df.T), print('='*50)

print(df[:2]), print('='*50)
print(df[:2].T)
```

       A  B   C   D
    0  0  1   2   3
    1  4  5   6   7
    2  8  9  10  11
    ==================================================
       0  1   2
    A  0  4   8
    B  1  5   9
    C  2  6  10
    D  3  7  11
    ==================================================
       A  B  C  D
    0  0  1  2  3
    1  4  5  6  7
    ==================================================
       0  1
    A  0  4
    B  1  5
    C  2  6
    D  3  7
    


```python
# 행과 열의 기본 처리

# 데이터 정렬 및 산술 연산 (넘파이 함수들과 데이터프레임 연동)

df3 = pd.DataFrame(np.arange(12).reshape(3,4), columns = ['A', 'B', 'C', 'D'])
print(df3), print('='*50)

print(np.exp(df3)), print('='*50)

np3 = np.asarray(df3)
print(np3)
```

       A  B   C   D
    0  0  1   2   3
    1  4  5   6   7
    2  8  9  10  11
    ==================================================
                 A            B             C             D
    0     1.000000     2.718282      7.389056     20.085537
    1    54.598150   148.413159    403.428793   1096.633158
    2  2980.957987  8103.083928  22026.465795  59874.141715
    ==================================================
    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]]
    


```python
# 인덱스 관련 객체

# 인덱스 객체
# Index 생성자를 사용하여 정수 리스트를 기반으로 ind라는 Index 객체를 생성
ind = pd.Index([1, 3, 5, 7, 9, 11])

print(ind), print('='*50)       # 생성된 Index 객체 출력
print(ind[1]), print('='*50)    # Index 객체의 특정 요소 접근 (NumPy 배열과 유사하게 인덱싱)
print(ind[::2]), print('='*50)  # Index 객체 슬라이싱 (NumPy 배열과 동일한 슬라이싱 방식)
print(ind.size, ind.shape, ind.ndim, ind.dtype) # Index 객체의 속성 확인
```

    Index([3, 3, 5, 7, 9, 11], dtype='int64')
    ==================================================
    3
    ==================================================
    Index([3, 5, 9], dtype='int64')
    ==================================================
    6 (6,) 1 int64
    


```python
# 인덱스 관련 객체

# 참고: Index는 중복된 값을 가질 수 있음
s = pd.Series([10, 20, 30, 40, 50], index=[1, 1, 3, 5, 5])
print(s)
print(s[1])  # 인덱스 1을 가진 모든 행 반환
```

    1    10
    1    20
    3    30
    5    40
    5    50
    dtype: int64
    1    10
    1    20
    dtype: int64
    


```python
# 인덱스 관련 객체

# pandas.Index 클래스
indA = pd.Index([1, 3, 5])
print(indA), print('='*50)

indB = pd.Index(list('abc'))
print(indB), print('='*50)

indC = indA.append(indB)
print(indC), print('='*50)

indD = indA.difference(indB)
print(indD)
```

    Index([1, 3, 5], dtype='int64')
    ==================================================
    Index(['a', 'b', 'c'], dtype='object')
    ==================================================
    Index([1, 3, 5, 'a', 'b', 'c'], dtype='object')
    ==================================================
    Index([1, 3, 5], dtype='int64')
    


```python
# 인덱스 관련 객체

# pandas.RangeIndex 클래스
df = pd.DataFrame(np.arange(12).reshape(2,6), columns = list('ABCDEF'))

print(df), print('='*50)

# df.index는 명시적인 인덱스가 제공되지 않았기 때문에
# Pandas가 자동으로 생성한 RangeIndex 객체를 보여줌
print(df.index)
```

       A  B  C  D   E   F
    0  0  1  2  3   4   5
    1  6  7  8  9  10  11
    ==================================================
    RangeIndex(start=0, stop=2, step=1)
    


```python
# 인덱스 관련 객체

# pandas.CategoricalIndex 클래스
s1 = pd.Series(['ha', 'hi']*1000)

print(s1), print('='*50)
# 각 문자열 'ha'와 'hi'는 여러 번 반복되어 저장되므로 메모리 사용량이 큼
print(s1.nbytes), print('='*50)

s2 = s1.astype('category')
print(s2), print('='*50)
# s2의 각 요소는 'ha','hi' 대신 해당 범주를 가리키는 짧은 정수 코드(0 또는 1)를 저장
print(s2.nbytes)
```

    0       ha
    1       hi
    2       ha
    3       hi
    4       ha
            ..
    1995    hi
    1996    ha
    1997    hi
    1998    ha
    1999    hi
    Length: 2000, dtype: object
    ==================================================
    16000
    ==================================================
    0       ha
    1       hi
    2       ha
    3       hi
    4       ha
            ..
    1995    hi
    1996    ha
    1997    hi
    1998    ha
    1999    hi
    Length: 2000, dtype: category
    Categories (2, object): ['ha', 'hi']
    ==================================================
    2016
    


```python
# 인덱스 관련 객체

# pandas.Categorical 클래스
s1 = pd.Categorical([1, 2, 3, 1, 2, 3])

print(s1), print('='*50)
print(type(s1), s1.dtype)

s2 = pd.Categorical(['a', 'b', 'c', 'a', 'b', 'c'])

print(s2), print('='*50)
print(type(s2), s2.dtype)

# ordered=True: 범주 순서에 따라 정렬되며 최소값과 최대값을 가질 수 있음
s3 = pd.Categorical(['a','b','c','a','b','c'],
                    ordered = True)

print(s3), print('='*50)
print(s3.min(), s3.max())

s4 = pd.Categorical(['a','b','c','a','b','c'],
                    ordered = True, categories = ['c','b','a'])

print(s4), print('='*50)
print(s4.min(), s4.max())
```

    [1, 2, 3, 1, 2, 3]
    Categories (3, int64): [1, 2, 3]
    ==================================================
    <class 'pandas.core.arrays.categorical.Categorical'> category
    ['a', 'b', 'c', 'a', 'b', 'c']
    Categories (3, object): ['a', 'b', 'c']
    ==================================================
    <class 'pandas.core.arrays.categorical.Categorical'> category
    ['a', 'b', 'c', 'a', 'b', 'c']
    Categories (3, object): ['a' < 'b' < 'c']
    ==================================================
    a c
    ['a', 'b', 'c', 'a', 'b', 'c']
    Categories (3, object): ['c' < 'b' < 'a']
    ==================================================
    c a
    


```python
# 인덱스 관련 객체

# 범주를 가지는 시리즈나 데이터프레임의 열에서 범주형 데이터를 생성하는 방법
# 1. 시리즈를 구성할 때 dtype = 'category'를 명시하는 방법
s1 = pd.Series(['a','b','c','a'], dtype='category')
print(s1), print('='*50)


# 2. 데이터프레임 생성자에 dtype = 'category'를 명시하는 방법
df2 = pd.DataFrame({'A': list('abca'), 'B': list('bccd')}, dtype='category')
print(df2.dtypes), print('-'*50)
print(df2), print('-'*50)
print(df2['A']), print('-'*50)
print(df2['B']), print('='*50)


# 3. 시리즈나 데이터프레임의 열을 category dtype으로 변환하는 방법
df = pd.DataFrame({'A': ['a','b','c','a']})

print(df), print('-'*50)
df['B'] = df['A'].astype('category')
print(df.dtypes), print('-'*50)
print(df), print('='*50)


# 4. DataFrame.astype()을 사용하여 데이터프레임의 모든 열을 한꺼번에 범주형으로 변환하는 방법
df3 = pd.DataFrame({'A': list('abca'), 'B': list('bccd')})
df_cat = df3.astype('category')

print(df_cat.dtypes), print('-'*50)
print(df_cat)
```

    0    a
    1    b
    2    c
    3    a
    dtype: category
    Categories (3, object): ['a', 'b', 'c']
    ==================================================
    A    category
    B    category
    dtype: object
    --------------------------------------------------
       A  B
    0  a  b
    1  b  c
    2  c  c
    3  a  d
    --------------------------------------------------
    0    a
    1    b
    2    c
    3    a
    Name: A, dtype: category
    Categories (3, object): ['a', 'b', 'c']
    --------------------------------------------------
    0    b
    1    c
    2    c
    3    d
    Name: B, dtype: category
    Categories (3, object): ['b', 'c', 'd']
    ==================================================
       A
    0  a
    1  b
    2  c
    3  a
    --------------------------------------------------
    A      object
    B    category
    dtype: object
    --------------------------------------------------
       A  B
    0  a  a
    1  b  b
    2  c  c
    3  a  a
    ==================================================
    A    category
    B    category
    dtype: object
    --------------------------------------------------
       A  B
    0  a  b
    1  b  c
    2  c  c
    3  a  d
    


```python
# 인덱스 관련 객체

# MultiIndex 생성 방법 1: MultiIndex.from_arrays(arrays)
# 각 인덱스 레벨에 해당하는 배열 (리스트, NumPy 배열, Series)의 리스트를 입력으로 받음

arrays = [['IVE', 'IVE', 'AESPA', 'AESPA'],
          ['Wonyoung', 'Liz', 'Winter', 'Karina']]

multi_index_arrays = pd.MultiIndex.from_arrays(arrays, names=['그룹', '이름'])
print(multi_index_arrays)
```

    MultiIndex([(  'IVE', 'Wonyoung'),
                (  'IVE',      'Liz'),
                ('AESPA',   'Winter'),
                ('AESPA',   'Karina')],
               names=['그룹', '이름'])
    


```python
# 인덱스 관련 객체

# MultiIndex 생성 방법 2: MultiIndex.from_product(iterables)
# 각 인덱스 레벨에 사용할 반복 가능한 객체 (리스트, 튜플 등)의 리스트를 입력으로 받음

groups = ['IVE', 'AESPA']
names = ['Wonyoung', 'Liz', 'Winter', 'Karina']
multi_index_product = pd.MultiIndex.from_product([groups, names], names=['그룹', '이름'])
print(multi_index_product)
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
    


```python
# 인덱스 관련 객체

# MultiIndex 생성 방법 3: MultiIndex.from_tuples(tuples)
# MultiIndex를 구성할 튜플의 리스트를 직접 입력으로 받음

tuples = [('IVE', 'Wonyoung'), ('IVE', 'Liz'), ('AESPA', 'Winter'), ('AESPA', 'Karina')]
multi_index_tuples = pd.MultiIndex.from_tuples(tuples, names=['그룹', '이름'])
print(multi_index_tuples)
```

    MultiIndex([(  'IVE', 'Wonyoung'),
                (  'IVE',      'Liz'),
                ('AESPA',   'Winter'),
                ('AESPA',   'Karina')],
               names=['그룹', '이름'])
    


```python
# 인덱스 관련 객체

# pandas.MultiIndex 클래스
arr = [[1, 1, 2, 2], ['red', 'blue', 'red', 'blue']]

mi_arr = pd.MultiIndex.from_arrays(arr, names=('number', 'color'))
print(mi_arr), print('='*50)

# 인덱스 생성자로 튜플의 리스트를 전달해 멀티인덱스를 반환
arr2 = [['ha','ha','hi','hi','ho','ho'], ['one','two','one','two','one','two']]

tuples = list(zip(*arr2))
print(tuples), print('-'*50)

mi_tuples = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
print(mi_tuples)
```

    MultiIndex([(1,  'red'),
                (1, 'blue'),
                (2,  'red'),
                (2, 'blue')],
               names=['number', 'color'])
    ==================================================
    [('ha', 'one'), ('ha', 'two'), ('hi', 'one'), ('hi', 'two'), ('ho', 'one'), ('ho', 'two')]
    --------------------------------------------------
    MultiIndex([('ha', 'one'),
                ('ha', 'two'),
                ('hi', 'one'),
                ('hi', 'two'),
                ('ho', 'one'),
                ('ho', 'two')],
               names=['first', 'second'])
    ==================================================
    1     1.324938
    3    -0.953207
    5     2.220743
    7    -0.002817
    9    -1.050837
    11    1.777364
    dtype: float64
    


```python
# 인덱스 관련 객체

# pandas.MultiIndex 클래스
# 시리즈나 데이터프레임에 배열의 리스트를 직접 전달하면 멀티인덱스를 자동 생성할 수 있음
arr = [np.array(['ha','ha','hi','hi','ho','ho']),
       np.array(['one','two','one','two','one','two'])]

ser = pd.Series(np.random.randn(6), index=arr)
print(ser), print('='*50)

df = pd.DataFrame(np.random.randn(6, 4), index=arr)
print(df), print('-'*50)
print(df.index), print('='*50)

# 튜플을 축의 행 라벨로 사용할 수도 있음
arr2 = [['ha','ha','hi','hi','ho','ho'], ['one','two','one','two','one','two']]
tuples = list(zip(*arr2))
print(tuples), print('-'*50)

ser2 = pd.Series(np.random.randn(6), index=tuples)
print(ser2)
```

    ha  one   -0.055870
        two    1.118636
    hi  one   -1.595272
        two   -2.828893
    ho  one   -0.277178
        two    2.413206
    dtype: float64
    ==================================================
                   0         1         2         3
    ha one  0.271772  0.123002  1.695759  0.905411
       two -0.214389 -1.543959  0.519001  0.835049
    hi one  0.138213  0.946325  0.704107 -1.885888
       two -1.033682  0.591992  0.582039 -0.549700
    ho one  0.384705  0.792180 -1.206690 -0.998811
       two -0.079801  0.059513 -0.776677  0.491232
    --------------------------------------------------
    MultiIndex([('ha', 'one'),
                ('ha', 'two'),
                ('hi', 'one'),
                ('hi', 'two'),
                ('ho', 'one'),
                ('ho', 'two')],
               )
    ==================================================
    [('ha', 'one'), ('ha', 'two'), ('hi', 'one'), ('hi', 'two'), ('ho', 'one'), ('ho', 'two')]
    --------------------------------------------------
    (ha, one)    0.902970
    (ha, two)    1.707262
    (hi, one)   -0.168082
    (hi, two)    0.132189
    (ho, one)    0.585904
    (ho, two)    1.448381
    dtype: float64
    

### 2. 판다스의 주요 기능


```python
# head()와 tail() 메소드의 적용

s1 = pd.Series(np.random.randn(1000))

print(s1.head()), print('-'*30)
print(s1.tail()), print('='*30)
'''
# 모든 행을 출력하도록 설정
pd.set_option('display.max_rows', None)
print(s1)

# 설정을 다시 기본값으로 되돌리고 싶다면..
# 최대 60개의 행까지만 화면에 표시
# 만약 데이터가 60행을 초과하면, 처음 몇 행과 마지막 몇 행만 보여주고
# 중간 부분은 생략 부호(...)로 표시
pd.set_option('display.max_rows', 60)
print(s1)
'''

ind = pd.date_range('1/1/2021', periods = 5)
df = pd.DataFrame(np.random.randn(5,3), index=ind, columns=['A','B','C'])

print(df.shape), print('-'*40)
print(df), print('='*40)
print(df[:2])
```

    0    0.456521
    1   -0.706182
    2    0.507156
    3    0.635036
    4   -0.125591
    dtype: float64
    ------------------------------
    995   -0.508586
    996   -0.113238
    997   -0.713856
    998    0.694662
    999    0.909094
    dtype: float64
    ==============================
    (5, 3)
    ----------------------------------------
                       A         B         C
    2021-01-01 -0.931134  0.537763  1.103203
    2021-01-02 -0.301499  0.452362  0.413956
    2021-01-03  0.203763 -0.042650 -1.306968
    2021-01-04 -1.665733 -1.472674 -1.574723
    2021-01-05  0.140433  0.671429 -1.209706
    ========================================
                       A         B         C
    2021-01-01 -0.931134  0.537763  1.103203
    2021-01-02 -0.301499  0.452362  0.413956
    


```python
# 판다스 객체 이진 연산

df2 = pd.DataFrame({'angles':[0, 3, 4], 'degrees':[360, 180, 360]},
                   index = ['circle', 'triangle', 'rectangle'])
print(df2), print('='*50)

# scalar 1을 더함
print(df2 + 1), print('='*50)
print(df2 - [1, 2]), print('='*50)
print(df2.sub([1, 2], axis='columns'))
```

               angles  degrees
    circle          0      360
    triangle        3      180
    rectangle       4      360
    ==================================================
               angles  degrees
    circle          1      361
    triangle        4      181
    rectangle       5      361
    ==================================================
               angles  degrees
    circle         -1      358
    triangle        2      178
    rectangle       3      358
    ==================================================
               angles  degrees
    circle         -1      358
    triangle        2      178
    rectangle       3      358
    


```python
# 판다스 객체 이진 연산

df = pd.DataFrame({'one': pd.Series(np.random.randn(2), index=['a','b']),
                   'two': pd.Series(np.random.randn(3), index=['a','b','c']),
                   'three': pd.Series(np.random.randn(2), index=['b','c'])})

print(df), print('-'*30)
print(df.iloc[1]), print('-'*30)
print(df['two']), print('='*30)

row = df.iloc[1]
col = df['two']

# 축의 값을 다르게 하여 sub()메소드 적용
# 'one' 열: df['one'] - row['one'], 'two' 열: df['two'] - row['two']
print(df.sub(row, axis='columns')), print('-'*30) # axis=1과 같음
# 'a' 행: df.loc['a'] - col['a'], 'b' 행: df.loc['b'] - col['b']
print(df.sub(col, axis='index'))    # axis=0과 같음
```

            one       two     three
    a  1.197873  0.965821       NaN
    b -0.358699 -1.202268  0.338031
    c       NaN  0.020510  0.564431
    ------------------------------
    one     -0.358699
    two     -1.202268
    three    0.338031
    Name: b, dtype: float64
    ------------------------------
    a    0.965821
    b   -1.202268
    c    0.020510
    Name: two, dtype: float64
    ==============================
            one       two   three
    a  1.556571  2.168088     NaN
    b  0.000000  0.000000  0.0000
    c       NaN  1.222777  0.2264
    ------------------------------
            one  two     three
    a  0.232052  0.0       NaN
    b  0.843569  0.0  1.540299
    c       NaN  0.0  0.543921
    


```python
# 판다스 객체 이진 연산

# 손실값 대체
d = {'one': [1., 2., np.nan], 'two': [3., 2., 1.], 'three': [np.nan, 1., 1.]}
df = pd.DataFrame(d, index = list('abc'))
print(df), print('='*50)

d1 = {'one': pd.Series([1., 2.], index = ['a','b']),
      'two': pd.Series([1., 1., 1.], index = ['a','b','c']),
      'three': pd.Series([2., 2., 2.], index = ['a','b','c'])}
df1 = pd.DataFrame(d1)
print(df1), print('='*50)

# 위의 df와 df1을 더할 때 두 데이터프레임에 손실 값이 있으면 손실 값을
# 0으로 바꾸어 연산 가능.
# 또한 fillna()를 사용하여 손실 값을 다른 값으로 변경 가능
print(df + df1), print('='*50)
print(df.add(df1, fill_value=0))
```

       one  two  three
    a  1.0  3.0    NaN
    b  2.0  2.0    1.0
    c  NaN  1.0    1.0
    ==================================================
       one  two  three
    a  1.0  1.0    2.0
    b  2.0  1.0    2.0
    c  NaN  1.0    2.0
    ==================================================
       one  two  three
    a  2.0  4.0    NaN
    b  4.0  3.0    3.0
    c  NaN  2.0    3.0
    ==================================================
       one  two  three
    a  2.0  4.0    2.0
    b  4.0  3.0    3.0
    c  NaN  2.0    3.0
    


```python
# 요약과 통계 연산

d = {'one': [1., 2., np.nan], 'two': [3., 2., 1.], 'three': [np.nan, 1., 1.]}
df = pd.DataFrame(d, index = list('abc'))
print(df), print('='*20)

print(df.mean(axis=0)), print('-'*20)
print(df.mean(1)), print('='*20)

# skipna 옵션: 손실 데이터를 배제할지 결정하는 옵션. 기본값은 True
print(df.sum(0, skipna=False)), print('-'*20)
# print(df.sum(axis=0, skipna=False)), print('-'*20) 위와 같음
print(df.sum(1, skipna=True))
```

       one  two  three
    a  1.0  3.0    NaN
    b  2.0  2.0    1.0
    c  NaN  1.0    1.0
    ====================
    one      1.5
    two      2.0
    three    1.0
    dtype: float64
    --------------------
    a    2.000000
    b    1.666667
    c    1.000000
    dtype: float64
    ====================
    one      NaN
    two      6.0
    three    NaN
    dtype: float64
    --------------------
    a    4.0
    b    5.0
    c    2.0
    dtype: float64
    


```python
# 요약과 통계 연산

# mean(), std(), sum()과 같은 넘파이 함수들은 시리즈 입력값에 있는 손실 값을 기본으로 제외
print(df), print('-'*20)
print(np.mean(df['one']))
```

       one  two  three
    a  1.0  3.0    NaN
    b  2.0  2.0    1.0
    c  NaN  1.0    1.0
    --------------------
    1.5
    


```python
# 요약과 통계 연산
print(df), print('='*20)

# std()의 인수 ddof: 데이터프레임에서의 ddof 기본값=1, 넘파이에서의 ddof 기본값=0
print(df.std()), print('-'*20)
print(df.std(axis=1)), print('-'*20)        # 데이터프레임에서의 기본값 ddof=1
print(np.std(df, axis=1)), print('-'*20)    # 넘파이에서의 기본값 ddof=0
print(np.std(df, ddof=1, axis=1)), print('-'*20)
print(df[['one', 'two', 'three']].std())
```

       one  two  three
    a  1.0  3.0    NaN
    b  2.0  2.0    1.0
    c  NaN  1.0    1.0
    ====================
    one      0.707107
    two      1.000000
    three    0.000000
    dtype: float64
    --------------------
    a    1.414214
    b    0.577350
    c    0.000000
    dtype: float64
    --------------------
    a    1.000000
    b    0.471405
    c    0.000000
    dtype: float64
    --------------------
    a    1.414214
    b    0.577350
    c    0.000000
    dtype: float64
    --------------------
    one      0.707107
    two      1.000000
    three    0.000000
    dtype: float64
    


```python
# 요약과 통계 연산

# 누적 합을 계산하는 cumsum() 메소드는 아래와 같이 적용
print(df), print('-'*20)
print(df.cumsum())
```

       one  two  three
    a  1.0  3.0    NaN
    b  2.0  2.0    1.0
    c  NaN  1.0    1.0
    --------------------
       one  two  three
    a  1.0  3.0    NaN
    b  3.0  5.0    1.0
    c  NaN  6.0    2.0
    


```python
# 요약과 통계 연산

# Series.nunique()
s1 = pd.Series(np.random.randn(500))
s1[20:500] = np.nan
s1[10:20] = 5
print(s1.nunique())

# 모든 행을 출력하도록 설정
pd.set_option('display.max_rows', None)
print(s1)

# 설정을 다시 기본값으로 되돌리고 싶다면
# pd.set_option('display.max_rows', 60)
```

    11
    0      0.964076
    1      0.075410
    2     -1.250127
    3      0.787937
    4     -1.939244
    5     -0.239285
    6      1.237872
    7      0.413716
    8      0.981545
    9      0.551829
    10     5.000000
    11     5.000000
    12     5.000000
    13     5.000000
    14     5.000000
    15     5.000000
    16     5.000000
    17     5.000000
    18     5.000000
    19     5.000000
    20          NaN
    21          NaN
    22          NaN
    23          NaN
    24          NaN
    25          NaN
    26          NaN
    27          NaN
    28          NaN
    29          NaN
    30          NaN
    31          NaN
    32          NaN
    33          NaN
    34          NaN
    35          NaN
    36          NaN
    37          NaN
    38          NaN
    39          NaN
    40          NaN
    41          NaN
    42          NaN
    43          NaN
    44          NaN
    45          NaN
    46          NaN
    47          NaN
    48          NaN
    49          NaN
    50          NaN
    51          NaN
    52          NaN
    53          NaN
    54          NaN
    55          NaN
    56          NaN
    57          NaN
    58          NaN
    59          NaN
    60          NaN
    61          NaN
    62          NaN
    63          NaN
    64          NaN
    65          NaN
    66          NaN
    67          NaN
    68          NaN
    69          NaN
    70          NaN
    71          NaN
    72          NaN
    73          NaN
    74          NaN
    75          NaN
    76          NaN
    77          NaN
    78          NaN
    79          NaN
    80          NaN
    81          NaN
    82          NaN
    83          NaN
    84          NaN
    85          NaN
    86          NaN
    87          NaN
    88          NaN
    89          NaN
    90          NaN
    91          NaN
    92          NaN
    93          NaN
    94          NaN
    95          NaN
    96          NaN
    97          NaN
    98          NaN
    99          NaN
    100         NaN
    101         NaN
    102         NaN
    103         NaN
    104         NaN
    105         NaN
    106         NaN
    107         NaN
    108         NaN
    109         NaN
    110         NaN
    111         NaN
    112         NaN
    113         NaN
    114         NaN
    115         NaN
    116         NaN
    117         NaN
    118         NaN
    119         NaN
    120         NaN
    121         NaN
    122         NaN
    123         NaN
    124         NaN
    125         NaN
    126         NaN
    127         NaN
    128         NaN
    129         NaN
    130         NaN
    131         NaN
    132         NaN
    133         NaN
    134         NaN
    135         NaN
    136         NaN
    137         NaN
    138         NaN
    139         NaN
    140         NaN
    141         NaN
    142         NaN
    143         NaN
    144         NaN
    145         NaN
    146         NaN
    147         NaN
    148         NaN
    149         NaN
    150         NaN
    151         NaN
    152         NaN
    153         NaN
    154         NaN
    155         NaN
    156         NaN
    157         NaN
    158         NaN
    159         NaN
    160         NaN
    161         NaN
    162         NaN
    163         NaN
    164         NaN
    165         NaN
    166         NaN
    167         NaN
    168         NaN
    169         NaN
    170         NaN
    171         NaN
    172         NaN
    173         NaN
    174         NaN
    175         NaN
    176         NaN
    177         NaN
    178         NaN
    179         NaN
    180         NaN
    181         NaN
    182         NaN
    183         NaN
    184         NaN
    185         NaN
    186         NaN
    187         NaN
    188         NaN
    189         NaN
    190         NaN
    191         NaN
    192         NaN
    193         NaN
    194         NaN
    195         NaN
    196         NaN
    197         NaN
    198         NaN
    199         NaN
    200         NaN
    201         NaN
    202         NaN
    203         NaN
    204         NaN
    205         NaN
    206         NaN
    207         NaN
    208         NaN
    209         NaN
    210         NaN
    211         NaN
    212         NaN
    213         NaN
    214         NaN
    215         NaN
    216         NaN
    217         NaN
    218         NaN
    219         NaN
    220         NaN
    221         NaN
    222         NaN
    223         NaN
    224         NaN
    225         NaN
    226         NaN
    227         NaN
    228         NaN
    229         NaN
    230         NaN
    231         NaN
    232         NaN
    233         NaN
    234         NaN
    235         NaN
    236         NaN
    237         NaN
    238         NaN
    239         NaN
    240         NaN
    241         NaN
    242         NaN
    243         NaN
    244         NaN
    245         NaN
    246         NaN
    247         NaN
    248         NaN
    249         NaN
    250         NaN
    251         NaN
    252         NaN
    253         NaN
    254         NaN
    255         NaN
    256         NaN
    257         NaN
    258         NaN
    259         NaN
    260         NaN
    261         NaN
    262         NaN
    263         NaN
    264         NaN
    265         NaN
    266         NaN
    267         NaN
    268         NaN
    269         NaN
    270         NaN
    271         NaN
    272         NaN
    273         NaN
    274         NaN
    275         NaN
    276         NaN
    277         NaN
    278         NaN
    279         NaN
    280         NaN
    281         NaN
    282         NaN
    283         NaN
    284         NaN
    285         NaN
    286         NaN
    287         NaN
    288         NaN
    289         NaN
    290         NaN
    291         NaN
    292         NaN
    293         NaN
    294         NaN
    295         NaN
    296         NaN
    297         NaN
    298         NaN
    299         NaN
    300         NaN
    301         NaN
    302         NaN
    303         NaN
    304         NaN
    305         NaN
    306         NaN
    307         NaN
    308         NaN
    309         NaN
    310         NaN
    311         NaN
    312         NaN
    313         NaN
    314         NaN
    315         NaN
    316         NaN
    317         NaN
    318         NaN
    319         NaN
    320         NaN
    321         NaN
    322         NaN
    323         NaN
    324         NaN
    325         NaN
    326         NaN
    327         NaN
    328         NaN
    329         NaN
    330         NaN
    331         NaN
    332         NaN
    333         NaN
    334         NaN
    335         NaN
    336         NaN
    337         NaN
    338         NaN
    339         NaN
    340         NaN
    341         NaN
    342         NaN
    343         NaN
    344         NaN
    345         NaN
    346         NaN
    347         NaN
    348         NaN
    349         NaN
    350         NaN
    351         NaN
    352         NaN
    353         NaN
    354         NaN
    355         NaN
    356         NaN
    357         NaN
    358         NaN
    359         NaN
    360         NaN
    361         NaN
    362         NaN
    363         NaN
    364         NaN
    365         NaN
    366         NaN
    367         NaN
    368         NaN
    369         NaN
    370         NaN
    371         NaN
    372         NaN
    373         NaN
    374         NaN
    375         NaN
    376         NaN
    377         NaN
    378         NaN
    379         NaN
    380         NaN
    381         NaN
    382         NaN
    383         NaN
    384         NaN
    385         NaN
    386         NaN
    387         NaN
    388         NaN
    389         NaN
    390         NaN
    391         NaN
    392         NaN
    393         NaN
    394         NaN
    395         NaN
    396         NaN
    397         NaN
    398         NaN
    399         NaN
    400         NaN
    401         NaN
    402         NaN
    403         NaN
    404         NaN
    405         NaN
    406         NaN
    407         NaN
    408         NaN
    409         NaN
    410         NaN
    411         NaN
    412         NaN
    413         NaN
    414         NaN
    415         NaN
    416         NaN
    417         NaN
    418         NaN
    419         NaN
    420         NaN
    421         NaN
    422         NaN
    423         NaN
    424         NaN
    425         NaN
    426         NaN
    427         NaN
    428         NaN
    429         NaN
    430         NaN
    431         NaN
    432         NaN
    433         NaN
    434         NaN
    435         NaN
    436         NaN
    437         NaN
    438         NaN
    439         NaN
    440         NaN
    441         NaN
    442         NaN
    443         NaN
    444         NaN
    445         NaN
    446         NaN
    447         NaN
    448         NaN
    449         NaN
    450         NaN
    451         NaN
    452         NaN
    453         NaN
    454         NaN
    455         NaN
    456         NaN
    457         NaN
    458         NaN
    459         NaN
    460         NaN
    461         NaN
    462         NaN
    463         NaN
    464         NaN
    465         NaN
    466         NaN
    467         NaN
    468         NaN
    469         NaN
    470         NaN
    471         NaN
    472         NaN
    473         NaN
    474         NaN
    475         NaN
    476         NaN
    477         NaN
    478         NaN
    479         NaN
    480         NaN
    481         NaN
    482         NaN
    483         NaN
    484         NaN
    485         NaN
    486         NaN
    487         NaN
    488         NaN
    489         NaN
    490         NaN
    491         NaN
    492         NaN
    493         NaN
    494         NaN
    495         NaN
    496         NaN
    497         NaN
    498         NaN
    499         NaN
    dtype: float64
    


```python
# 요약과 통계 연산

# describe()
s2 = pd.Series(np.random.randn(1000))
s2[::2] = np.nan
print(s2.describe()), print('-'*30)

df = pd.DataFrame(np.random.randn(1000,4), columns=['a','b','c','d'])
df.iloc[::2] = np.nan
print(df.describe()), print('-'*30)

# 출력에 포함할 특정 백분위수를 선택할 수 있음
print(s2.describe(percentiles=[0.05, 0.25, 0.75, 0.95]))
```

    count    500.000000
    mean      -0.023498
    std        0.999061
    min       -3.359483
    25%       -0.698030
    50%       -0.080115
    75%        0.745869
    max        2.687528
    dtype: float64
    ------------------------------
                    a           b           c           d
    count  500.000000  500.000000  500.000000  500.000000
    mean    -0.053265   -0.086127    0.004521   -0.063966
    std      0.999191    0.940253    1.006886    0.995312
    min     -3.259231   -2.765092   -4.357936   -3.761679
    25%     -0.768429   -0.720298   -0.608466   -0.702297
    50%     -0.036965   -0.052279    0.006832   -0.064997
    75%      0.633947    0.564432    0.672154    0.634819
    max      2.574205    2.342424    3.107050    2.975626
    ------------------------------
    count    500.000000
    mean      -0.023498
    std        0.999061
    min       -3.359483
    5%        -1.684938
    25%       -0.698030
    50%       -0.080115
    75%        0.745869
    95%        1.623334
    max        2.687528
    dtype: float64
    ------------------------------
    




    (None, None)




```python
# 요약과 통계 연산

# describe()
# 수치가 아닌 객체에 describe() 메소드를 적용하면, 유일값 수와 가장 빈번히 발생하는 값 요약
s3 = pd.Series(['a','a','b','c','c',np.nan,'c','d'])
print(s3.describe()), print('-'*30)

# 범주형과 수치가 혼합된 타입의 데이터프레임 객체에 describe() 메소드를 적용하면 수치로 이루어진 열만 반환
df = pd.DataFrame({'a': ['Yes','Yes','No','No'], 'b': range(4)})
print(df.describe()), print('-'*30)

# describe()에 include와 exclude (적용하지 않을 열 전달) 인수 적용 가능
print(df.describe(include=['object'])), print('-'*30)
print(df.describe(include=['number'])), print('-'*30)
print(df.describe(include='all'))
```

    count     7
    unique    4
    top       c
    freq      3
    dtype: object
    ------------------------------
                  b
    count  4.000000
    mean   1.500000
    std    1.290994
    min    0.000000
    25%    0.750000
    50%    1.500000
    75%    2.250000
    max    3.000000
    ------------------------------
              a
    count     4
    unique    2
    top     Yes
    freq      2
    ------------------------------
                  b
    count  4.000000
    mean   1.500000
    std    1.290994
    min    0.000000
    25%    0.750000
    50%    1.500000
    75%    2.250000
    max    3.000000
    ------------------------------
              a         b
    count     4  4.000000
    unique    2       NaN
    top     Yes       NaN
    freq      2       NaN
    mean    NaN  1.500000
    std     NaN  1.290994
    min     NaN  0.000000
    25%     NaN  0.750000
    50%     NaN  1.500000
    75%     NaN  2.250000
    max     NaN  3.000000
    


```python
# 요약과 통계 연산

# idxmax()와 idxmin() 메소드
s1 = pd.Series(np.random.randn(5))
print(s1), print('-'*20)

print(s1.idxmin(), s1.idxmax())

df = pd.DataFrame(np.random.randn(4,3), columns=['A', 'B', 'C'])
print(df), print('-'*30)
print(df.idxmin(axis=0)), print('-'*30)
print(df.idxmin()), print('-'*30)
print(df.idxmax(axis=1)), print('='*30)

# 최소값과 최대값에 일치하는 행이나 열이 다수라면 첫 번째로 일치하는 인덱스 반환
df1 = pd.DataFrame([2, 1, 1, 3, np.nan], columns=['A'], index=list('edcba'))
print(df1), print('-'*30)

print(df1['A'].idxmin())
```

    0   -0.746161
    1    0.074425
    2   -0.025729
    3    0.663635
    4    1.731401
    dtype: float64
    --------------------
    0 4
              A         B         C
    0 -1.248322 -1.619103  0.471988
    1 -1.662002  1.227801  1.003069
    2  0.569126  1.077331  1.275057
    3  1.123960  0.861832 -0.173069
    ------------------------------
    A    1
    B    0
    C    3
    dtype: int64
    ------------------------------
    A    1
    B    0
    C    3
    dtype: int64
    ------------------------------
    0    C
    1    B
    2    C
    3    A
    dtype: object
    ==============================
         A
    e  2.0
    d  1.0
    c  1.0
    b  3.0
    a  NaN
    ------------------------------
    d
    


```python
# 요약과 통계 연산

# value_counts() 메소드
data = np.random.randint(0, 7, size=30)
print(data), print('-'*60)

s1 = pd.Series(data)
print(s1.value_counts()), print('-'*30)

print(pd.value_counts(data))

```

    [5 3 5 5 5 0 4 2 4 5 4 5 2 5 5 4 3 2 2 3 4 5 1 3 2 6 1 6 5 4]
    ------------------------------------------------------------
    5    10
    4     6
    2     5
    3     4
    1     2
    6     2
    0     1
    Name: count, dtype: int64
    ------------------------------
    5    10
    4     6
    2     5
    3     4
    1     2
    6     2
    0     1
    Name: count, dtype: int64
    ------------------------------
    

    <ipython-input-50-cba9d4e31caa>:9: FutureWarning: pandas.value_counts is deprecated and will be removed in a future version. Use pd.Series(obj).value_counts() instead.
      print(pd.value_counts(data)), print('-'*30)
    




    (None, None)




```python
# 요약과 통계 연산

# cut() 함수
print(pd.cut(np.array([1, 7, 5, 4, 6, 3]), bins=3)), print('='*80)

# 인수 retbins=True는 구간 값을 반환
print(pd.cut(np.array([1, 7, 5, 4, 6, 3]), 3, retbins=True)), print('='*80)

# 구간에 특정 라벨 ('bad', 'medium', 'good')을 할당하여 labels 범주를 반환하는 예
print(pd.cut(np.array([1, 7, 5, 4, 6, 3]), 3, labels=['bad','medium','good'])), print('='*80)

# labels=False를 입력하면 범주형 시리즈나 정수 배열을 반환
print(pd.cut([0, 1, 1, 2], bins=4)), print('='*80)
print(pd.cut([0, 1, 1, 2], bins=4, labels=False)), print('='*80)

# cut()함수에 시리즈를 입력하면 categorical dtype인 시리즈를 반환
s1 = pd.Series(np.array([2,4,6,8,10]), index=['a','b','c','d','e'])
print(pd.cut(s1, bins=3))
```

    [(0.994, 3.0], (5.0, 7.0], (3.0, 5.0], (3.0, 5.0], (5.0, 7.0], (0.994, 3.0]]
    Categories (3, interval[float64, right]): [(0.994, 3.0] < (3.0, 5.0] < (5.0, 7.0]]
    ================================================================================
    ([(0.994, 3.0], (5.0, 7.0], (3.0, 5.0], (3.0, 5.0], (5.0, 7.0], (0.994, 3.0]]
    Categories (3, interval[float64, right]): [(0.994, 3.0] < (3.0, 5.0] < (5.0, 7.0]], array([0.994, 3.   , 5.   , 7.   ]))
    ================================================================================
    ['bad', 'good', 'medium', 'medium', 'good', 'bad']
    Categories (3, object): ['bad' < 'medium' < 'good']
    ================================================================================
    [(-0.002, 0.5], (0.5, 1.0], (0.5, 1.0], (1.5, 2.0]]
    Categories (4, interval[float64, right]): [(-0.002, 0.5] < (0.5, 1.0] < (1.0, 1.5] < (1.5, 2.0]]
    ================================================================================
    [0 1 1 3]
    ================================================================================
    a    (1.992, 4.667]
    b    (1.992, 4.667]
    c    (4.667, 7.333]
    d     (7.333, 10.0]
    e     (7.333, 10.0]
    dtype: category
    Categories (3, interval[float64, right]): [(1.992, 4.667] < (4.667, 7.333] < (7.333, 10.0]]
    


```python
# 요약과 통계 연산

# qcut() 함수
print(pd.qcut(range(5),4)), print('-'*70)

print(pd.qcut(range(5), 3, labels=['good','medium','bad'])), print('-'*70)

print(pd.qcut(range(5), 4, labels=False))
```

    [(-0.001, 1.0], (-0.001, 1.0], (1.0, 2.0], (2.0, 3.0], (3.0, 4.0]]
    Categories (4, interval[float64, right]): [(-0.001, 1.0] < (1.0, 2.0] < (2.0, 3.0] < (3.0, 4.0]]
    ----------------------------------------------------------------------
    ['good', 'good', 'medium', 'bad', 'bad']
    Categories (3, object): ['good' < 'medium' < 'bad']
    ----------------------------------------------------------------------
    [0 0 1 2 3]
    


```python
# 요약과 통계 연산

# cut()과 qcut() 함수 차이
random_sample = np.random.randn(25)
print(pd.cut(random_sample, 5).value_counts())

print(pd.qcut(random_sample, 5).value_counts())
```

    (-1.738, -1.1]     1
    (-1.1, -0.466]     0
    (-0.466, 0.169]    3
    (0.169, 0.804]     0
    (0.804, 1.439]     1
    Name: count, dtype: int64
    (-1.736, -0.436]    2
    (-0.436, -0.105]    1
    (-0.105, 0.122]     1
    (0.122, 1.439]      1
    Name: count, dtype: int64
    
