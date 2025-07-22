### 11주차 강의


```python
# 요약과 통계 연산

d = {'one': [1., 2., np.nan], 'two': [3., 2., 1.], 'three': [np.nan, 1., 1.]}
df = pd.DataFrame(d, index = list('abc'))
print(df), print('='*20)

print(df.mean(axis=0)), print('-'*20)
print(df.mean(1)), print('='*20)

# np.mean(), np.std(), np.sum()과 같은 넘파이 함수들은 손실 값을 기본으로 제외
print(np.mean(df['one'])), print('='*20)

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
    1.5
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
print(df), print('='*20)

# std()의 인수 ddof: 데이터프레임에서의 ddof 기본값=1, 넘파이에서의 ddof 기본값=0
print(df.std()), print('-'*20)
print(df.std(axis=1)), print('-'*20)        # 데이터프레임에서의 기본값 ddof=1 (표본 표준편차)
print(np.std(df, axis=1)), print('-'*20)    # 넘파이에서의 기본값 ddof=0 (모집단 표준편차)
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

# cumsum() 메소드(누적 합): 기본으로 NaN 무시, but 결과배열에서는 NaN 유지
print(df), print('-'*20)
print(df.cumsum()), print('-'*20)
print(df.cumsum(axis=1)), print('-'*20)
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
    --------------------
       one  two  three
    a  1.0  4.0    NaN
    b  2.0  4.0    5.0
    c  NaN  1.0    2.0
    --------------------
    one      2.0
    two      6.0
    three    1.0
    dtype: float64
    


```python
# 요약과 통계 연산

# Series.nunique(): 고유한 값의 개수를 반
s1 = pd.Series(np.random.randn(500))
s1[20:500] = np.nan
s1[10:20] = 5
print(s1.nunique()), print('-'*20)
print(s1), print('-'*20)

print(np.unique(s1))
print(np.unique(s1).size)
# # 출력 개수 늘리기 (예: 20개까지 출력)
# pd.set_option('display.max_rows', 40)
# print(s1)

# # 모든 행을 출력하도록 설정
# pd.set_option('display.max_rows', None)
# print(s1)

```

    11
    --------------------
    0      0.181285
    1     -2.050039
    2      1.708496
    3     -1.036937
    4      0.515096
             ...   
    495         NaN
    496         NaN
    497         NaN
    498         NaN
    499         NaN
    Length: 500, dtype: float64
    --------------------
    [-2.05003864 -1.0369371  -1.00410389 -0.74040423 -0.61226158  0.13634265
      0.18128508  0.51509615  0.7713619   1.70849561  5.                 nan]
    12
    


```python
# 요약과 통계 연산

# describe()
s2 = pd.Series(np.random.randn(1000))
s2[::2] = np.nan
print(s2.describe()), print('-'*30)

# 출력에 포함할 특정 백분위수를 선택할 수 있음
print(s2.describe(percentiles=[0.05, 0.25, 0.75, 0.95])), print('-'*30)


df = pd.DataFrame(np.random.randn(1000,4), columns=['a','b','c','d'])
df.iloc[::2] = np.nan
print(df.describe())
```

    count    500.000000
    mean      -0.016282
    std        0.925681
    min       -2.853213
    25%       -0.681221
    50%       -0.048098
    75%        0.611231
    max        2.633256
    dtype: float64
    ------------------------------
    count    500.000000
    mean      -0.016282
    std        0.925681
    min       -2.853213
    5%        -1.403101
    25%       -0.681221
    50%       -0.048098
    75%        0.611231
    95%        1.484673
    max        2.633256
    dtype: float64
    ------------------------------
                    a           b           c           d
    count  500.000000  500.000000  500.000000  500.000000
    mean    -0.014791   -0.060565    0.052240    0.078893
    std      1.070563    0.944839    1.024369    1.038375
    min     -3.152755   -2.753698   -2.920871   -3.115698
    25%     -0.746640   -0.755098   -0.603881   -0.607735
    50%     -0.027622   -0.015279    0.104951    0.071106
    75%      0.694156    0.599519    0.671132    0.798520
    max      3.148538    2.643058    3.195628    3.101007
    


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

print(s1.idxmin(), s1.idxmax()) # 최소값과 최대값의 인덱스 반환

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

print(pd.value_counts(data))    # FutureWaring

## 참고만...
# 넘파이의 np.unique with return_counts=True
unique_values, counts = np.unique(s1, return_counts=True)

print("고유한 값:", unique_values)
print("빈도수:", counts)

# 빈도수를 기준으로 내림차순 정렬하기 위한 인덱스 얻기
sorted_indices = np.argsort(counts)[::-1]

# 정렬된 고유한 값과 빈도수 얻기
sorted_unique_values = unique_values[sorted_indices]
sorted_counts = counts[sorted_indices]

print("정렬된 고유한 값 (빈도수 내림차순):", sorted_unique_values)
print("정렬된 빈도수 (내림차순):", sorted_counts)
```

    [3 4 0 3 4 2 0 3 5 4 6 5 0 1 6 5 4 6 5 0 2 3 4 3 0 5 5 0 6 6]
    ------------------------------------------------------------
    0    6
    5    6
    3    5
    4    5
    6    5
    2    2
    1    1
    Name: count, dtype: int64
    ------------------------------
    0    6
    5    6
    3    5
    4    5
    6    5
    2    2
    1    1
    Name: count, dtype: int64
    고유한 값: [0 1 2 3 4 5 6]
    빈도수: [6 1 2 5 5 6 5]
    정렬된 고유한 값 (빈도수 내림차순): [5 0 6 4 3 2 1]
    정렬된 빈도수 (내림차순): [6 6 5 5 5 2 1]
    

    <ipython-input-11-9a12170dc8c5>:10: FutureWarning: pandas.value_counts is deprecated and will be removed in a future version. Use pd.Series(obj).value_counts() instead.
      print(pd.value_counts(data))    # FutureWaring
    


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

    (-1.821, -1.228]      5
    (-1.228, -0.637]      0
    (-0.637, -0.0474]     2
    (-0.0474, 0.543]     10
    (0.543, 1.133]        8
    Name: count, dtype: int64
    (-1.819, -0.614]    5
    (-0.614, 0.169]     5
    (0.169, 0.385]      5
    (0.385, 0.71]       5
    (0.71, 1.133]       5
    Name: count, dtype: int64
    


```python
# 함수 적용

# 1. 테이블 전체 함수 적용: pipe()
data = pd.DataFrame([[1, 1, 1,], [2, 2, 2], [3, 3, 3]],
                    index=['A','B','C'], columns=['one','two','three'])

print(data), print('='*50)

# 사용자 정의 함수를 사용한 사칙 연산
def add(data, arg):
    return data + arg

def div(data, arg):
    return data / arg

def mul(data, arg):
    return data * arg

def sub(data, arg):
    return data - arg

# data 객체에 사용자 정의함수를 pipe()메소드를 이용해 체인 형태로 묶어 실행
result = (data.pipe(add, arg=2)     # data에 2를 더하기
            .pipe(div, arg=3)      # 위 결과에 3으로 나누기
            .pipe(mul, arg=5)      # 위 결과에 5를 곱하기
            .pipe(sub, arg=1))     # 위 결과에서 1을 빼기
print(result)
```

       one  two  three
    A    1    1      1
    B    2    2      2
    C    3    3      3
    ==================================================
            one       two     three
    A  4.000000  4.000000  4.000000
    B  5.666667  5.666667  5.666667
    C  7.333333  7.333333  7.333333
    


```python
# 함수 적용

# 2. 행 또는 열 단위의 함수를 적용: apply()
data = [{'one': 1.0, 'two': 1.2},
        {'one': 0.5, 'two': 1.1, 'three': 0.7},
        {'one': 0.7, 'two': 0.9, 'three': -1.6},
                    {'two':1.4, 'three': -1.2}]

df = pd.DataFrame(data)
print(df), print('='*30)

# apply() 메소드를 사용해 np.mean을 적용하면 열을 중심으로 평균값 구함
# 인수 axis=1을 설정하면 행을 기준으로 연산함
print(df.apply(np.mean)), print('='*30)
print(df.apply(np.mean, axis=1)), print('='*30)

# lambda 함수 적용하여 최대값과 최소값의 차이 구함
print(df.apply(lambda x: x.max() - x.min())), print('='*30)

# np.cumsum을 적용하여 누적합 구함
print(df.apply(np.cumsum)), print('='*30)

# np.exp를 적용하여 지수를 계산
print(df.apply(np.exp))
```

       one  two  three
    0  1.0  1.2    NaN
    1  0.5  1.1    0.7
    2  0.7  0.9   -1.6
    3  NaN  1.4   -1.2
    ==============================
    one      0.733333
    two      1.150000
    three   -0.700000
    dtype: float64
    ==============================
    0    1.100000
    1    0.766667
    2    0.000000
    3    0.100000
    dtype: float64
    ==============================
    one      0.5
    two      0.5
    three    2.3
    dtype: float64
    ==============================
       one  two  three
    0  1.0  1.2    NaN
    1  1.5  2.3    0.7
    2  2.2  3.2   -0.9
    3  NaN  4.6   -2.1
    ==============================
            one       two     three
    0  2.718282  3.320117       NaN
    1  1.648721  3.004166  2.013753
    2  2.013753  2.459603  0.201897
    3       NaN  4.055200  0.301194
    


```python
# 일반 함수 vs. lambda 함수

# 일반 함수
def add(x, y):
    return x + y

# lambda 함수
# add_lambda는 함수 이름이 아니고, lambda 익명함수를 참조하는 변수 이름임
add_lambda = lambda x, y: x + y

print(add(3, 5))         # 8
print(add_lambda(3, 5))  # 8

```

    8
    8
    


```python
# 함수 적용

# 3. Aggregation API: agg()
df = pd.DataFrame([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9],
                   [np.nan, np.nan, np.nan]], columns=['A', 'B', 'C'])

print(df), print('='*30)

# agg()에 축을 명시하지 않으면 각 열에 각기 다른 종합 연산을 실행
print(df.agg(['sum', 'min'])), print('='*30)
print(df.agg(['sum', 'min'])['A']), print('='*30)

# axis='columns'인 경우, 행에 연산을 실행
print(df.agg(['sum', 'min'], axis='columns')), print('='*30)
print(df.agg(['sum', 'min'], axis=1))

```

         A    B    C
    0  1.0  2.0  3.0
    1  4.0  5.0  6.0
    2  7.0  8.0  9.0
    3  NaN  NaN  NaN
    ==============================
            A     B     C
    sum  12.0  15.0  18.0
    min   1.0   2.0   3.0
    ==============================
    sum    12.0
    min     1.0
    Name: A, dtype: float64
    ==============================
        sum  min
    0   6.0  1.0
    1  15.0  4.0
    2  24.0  7.0
    3   0.0  NaN
    ==============================
        sum  min
    0   6.0  1.0
    1  15.0  4.0
    2  24.0  7.0
    3   0.0  NaN
    


```python
# 함수 적용

# 3. Aggregation API: agg()
adf = pd.DataFrame(np.random.randn(6,3), columns=['A','B','C'],
                  index=pd.date_range('7/1/2021', periods=6))

adf.iloc[2:4] = np.nan
print(adf), print('='*40)

# 단일 함수 사용시 apply() 메소드를 사용하는 방법과 동일
print(adf.agg(np.sum)), print('='*40) # FutureWarning

print(adf.agg('sum')), print('='*40)

print(adf.sum()), print('='*40)

# 시리즈에 단일 종합 연산을 실행하면 스칼라 값을 반환
print(adf.A.agg('sum'))
```

                       A         B         C
    2021-07-01 -0.507650 -0.178433  0.628265
    2021-07-02  1.188533 -0.100612  0.899260
    2021-07-03       NaN       NaN       NaN
    2021-07-04       NaN       NaN       NaN
    2021-07-05  0.018694  0.456312  0.823443
    2021-07-06  1.442214 -1.156142 -0.088366
    ========================================
    A    2.141792
    B   -0.978875
    C    2.262602
    dtype: float64
    ========================================
    A    2.141792
    B   -0.978875
    C    2.262602
    dtype: float64
    ========================================
    A    2.141792
    B   -0.978875
    C    2.262602
    dtype: float64
    ========================================
    2.1417919118391087
    

    <ipython-input-17-5ae4ee4633af>:11: FutureWarning: The provided callable <function sum at 0x795018604cc0> is currently using DataFrame.sum. In a future version of pandas, the provided callable will be used directly. To keep current behavior pass the string "sum" instead.
      print(adf.agg(np.sum)), print('='*40) # FutureWarning
    


```python
# 함수 적용

# 3. Aggregation API: agg()

# 복수의 함수로 종합 연산을 실행할 때 여러 인수를 리스트로 전달할 수 있음
print(adf.agg(['sum'])), print('='*40)

# 복수의 함수를 사용하면, 복수의 행을 반환
print(adf.agg(['sum', 'mean'])), print('='*40)

# lambda 함수를 입력하면 <lambda> 이름의 행을 반환
print(adf.A.agg(['sum', lambda x: x.mean()])), print('='*40)

# 사용자 정의 함수를 입력하면 행에 그 함수의 이름을 반환
def mymean(x):
    return x.mean()

print(adf.A.agg(['sum', mymean])), print('='*40)

# DataFrame.agg()로 각 열에 함수 이름을 딕셔너리형으로 전달하면 각 열에 해당 함수를 적용함
print(adf.agg({'A': 'mean', 'B': 'sum'})), print('='*40)

# 열 라벨에 유사 리스트로 함수를 전달하면 데이터프레임 형태로 출력됨
print(adf.agg({'A': ['mean', 'min'], 'B': 'sum'}))
```

                A         B         C
    sum  0.845429 -1.772702 -2.159888
    ========================================
                 A         B         C
    sum   0.845429 -1.772702 -2.159888
    mean  0.211357 -0.443175 -0.539972
    ========================================
    sum         0.845429
    <lambda>    0.211357
    Name: A, dtype: float64
    ========================================
    sum       0.845429
    mymean    0.211357
    Name: A, dtype: float64
    ========================================
    A    0.211357
    B   -1.772702
    dtype: float64
    ========================================
                 A         B
    mean  0.211357       NaN
    min  -1.448441       NaN
    sum        NaN -1.772702
    


```python
# 함수 적용

# 3. Aggregation API: transform()

df = pd.DataFrame({'A': range(3), 'B': range(1, 4)})

print(df), print('-'*30)

result = df.transform(lambda x: x+1)
print(result)
```

       A  B
    0  0  1
    1  1  2
    2  2  3
    ------------------------------
       A  B
    0  1  2
    1  2  3
    2  3  4
    


```python
# 함수 적용

# 3. Aggregation API: transform()
# transform() 메소드에는 넘파이 함수, 문자열인 함수이름 or
# 사용자 정의함수를 입력할 수 있으며 전체 프레임을 반환함
adf = pd.DataFrame(np.random.randn(6,3), columns=['A','B','C'],
                  index=pd.date_range('7/1/2021', periods=6))
adf.iloc[2:4] = np.nan
print(adf), print('='*40)

# np.abs는 집계함수가 아니고 요소함수라서 agg()의 DataFrame의 형태가 유지됨
print(adf.agg(np.abs)), print('-'*40)       # 아래와 같은 결과
print(adf.transform(np.abs)), print('='*40) # 위와 같은 결과

# 시리즈에 함수 하나를 입력하면 단일 시리즈를 반환
print(adf.A.transform(np.abs))
```

                       A         B         C
    2021-07-01 -0.581825 -0.344163  1.743990
    2021-07-02  0.181776  0.436731  0.505913
    2021-07-03       NaN       NaN       NaN
    2021-07-04       NaN       NaN       NaN
    2021-07-05 -0.140522 -1.231325  0.130688
    2021-07-06 -0.330145  0.447075  1.109982
    ========================================
                       A         B         C
    2021-07-01  0.581825  0.344163  1.743990
    2021-07-02  0.181776  0.436731  0.505913
    2021-07-03       NaN       NaN       NaN
    2021-07-04       NaN       NaN       NaN
    2021-07-05  0.140522  1.231325  0.130688
    2021-07-06  0.330145  0.447075  1.109982
    ----------------------------------------
                       A         B         C
    2021-07-01  0.581825  0.344163  1.743990
    2021-07-02  0.181776  0.436731  0.505913
    2021-07-03       NaN       NaN       NaN
    2021-07-04       NaN       NaN       NaN
    2021-07-05  0.140522  1.231325  0.130688
    2021-07-06  0.330145  0.447075  1.109982
    ========================================
    2021-07-01    0.581825
    2021-07-02    0.181776
    2021-07-03         NaN
    2021-07-04         NaN
    2021-07-05    0.140522
    2021-07-06    0.330145
    Freq: D, Name: A, dtype: float64
    


```python
# 함수 적용

# 3. Aggregation API: transform()
# transform() 메소드에는 넘파이 함수, 문자열인 함수이름 or
# 사용자 정의함수를 입력할 수 있으며 전체 프레임을 반환함
adf = pd.DataFrame(np.random.randn(6,3), columns=['A','B','C'],
                  index=pd.date_range('7/1/2021', periods=6))
adf.iloc[2:4] = np.nan
print(adf), print('='*40)

# 복수의 함수를 전달하면 멀티인덱스를 가진 열로 이루어진 데이터프레임을 반환
# 첫번째 레벨은 원래 데이터프레임 열 이름이고, 두번째 레벨은 변환하는 함수의 이름
print(adf.transform([np.abs, lambda x: x+1])), print('='*70)

# 복수의 함수를 시리즈로 입력하면 데이터프레임을 반환
print(adf.A.transform([np.abs, lambda x: x+1])), print('='*40)

# 함수들로 이루어진 딕셔너리를 전달하면 각 열마다 연산을 적용
print(adf.transform({'A': np.abs, 'B': lambda x: x+1})), print('='*40)

# 리스트가 있는 딕셔너리를 전달하면 각 열에 함수를 호출하여 연산을 실행하며
# 멀티인덱스를 가진 데이터프레임을 생성 (음수에 sqrt적용하면 NaN 출력)
print(adf.transform({'A': np.abs, 'B': [lambda x: x+1, 'sqrt']}))
```

                       A         B         C
    2021-07-01  0.772104 -0.197569  0.654297
    2021-07-02  1.406169 -0.006265  1.427475
    2021-07-03       NaN       NaN       NaN
    2021-07-04       NaN       NaN       NaN
    2021-07-05 -0.219543 -0.049936 -0.741405
    2021-07-06  0.066093 -1.163254  0.514730
    ========================================
                       A                   B                   C          
                absolute  <lambda>  absolute  <lambda>  absolute  <lambda>
    2021-07-01  0.772104  1.772104  0.197569  0.802431  0.654297  1.654297
    2021-07-02  1.406169  2.406169  0.006265  0.993735  1.427475  2.427475
    2021-07-03       NaN       NaN       NaN       NaN       NaN       NaN
    2021-07-04       NaN       NaN       NaN       NaN       NaN       NaN
    2021-07-05  0.219543  0.780457  0.049936  0.950064  0.741405  0.258595
    2021-07-06  0.066093  1.066093  1.163254 -0.163254  0.514730  1.514730
    ======================================================================
                absolute  <lambda>
    2021-07-01  0.772104  1.772104
    2021-07-02  1.406169  2.406169
    2021-07-03       NaN       NaN
    2021-07-04       NaN       NaN
    2021-07-05  0.219543  0.780457
    2021-07-06  0.066093  1.066093
    ========================================
                       A         B
    2021-07-01  0.772104  0.802431
    2021-07-02  1.406169  0.993735
    2021-07-03       NaN       NaN
    2021-07-04       NaN       NaN
    2021-07-05  0.219543  0.950064
    2021-07-06  0.066093 -0.163254
    ========================================
                       A         B     
                absolute  <lambda> sqrt
    2021-07-01  0.772104  0.802431  NaN
    2021-07-02  1.406169  0.993735  NaN
    2021-07-03       NaN       NaN  NaN
    2021-07-04       NaN       NaN  NaN
    2021-07-05  0.219543  0.950064  NaN
    2021-07-06  0.066093 -0.163254  NaN
    

    /usr/local/lib/python3.11/dist-packages/pandas/core/arraylike.py:399: RuntimeWarning: invalid value encountered in sqrt
      result = getattr(ufunc, method)(*inputs, **kwargs)
    


```python
# 함수 적용

# 4. 요소 단위 함수 적용

# 데이터프레임 df에 lambda 함수를 적용하거나 제곱 연산을 실행한 예제
print(df.applymap(lambda x: x**2)), print('-'*30)
print(df**2), print('='*30)


s1 = pd.Series(['cat', 'dog', np.nan, 'rabbit'])
print(s1), print('-'*30)

# 인수가 딕셔너리형일 때, 딕셔너리 내에 키로써 존재하지 않는 시리즈 값들은 NaN으로 반환
# map( ) 메소드는 딕셔너리, 시리즈, 사용자 정의 함수를 인수로 가질 수 있음
result = s1.map({'cat': 'kitten', 'dog': 'puppy'})
print(result), print('-'*30)
print(s1.map('I am a {}'.format)), print('-'*30)

print(s1.map('I am a {}'.format, na_action='ignore'))
```

           0      1
    0  1.000  2.120
    1  3.356  4.567
    ------------------------------
           0      1
    0  1.000  2.120
    1  3.356  4.567
       0  1
    0  3  4
    1  5  5
    ------------------------------
               0          1
    0   1.000000   4.494400
    1  11.262736  20.857489
    ------------------------------
               0          1
    0   1.000000   4.494400
    1  11.262736  20.857489
    ==============================
    0       cat
    1       dog
    2       NaN
    3    rabbit
    dtype: object
    ------------------------------
    0    kitten
    1     puppy
    2       NaN
    3       NaN
    dtype: object
    ------------------------------
    0       I am a cat
    1       I am a dog
    2       I am a nan
    3    I am a rabbit
    dtype: object
    ------------------------------
    0       I am a cat
    1       I am a dog
    2              NaN
    3    I am a rabbit
    dtype: object
    

    <ipython-input-18-d972b08495cc>:10: FutureWarning: DataFrame.applymap has been deprecated. Use DataFrame.map instead.
      print(df.applymap(lambda x: len(str(x)))), print('-'*30)
    <ipython-input-18-d972b08495cc>:13: FutureWarning: DataFrame.applymap has been deprecated. Use DataFrame.map instead.
      print(df.applymap(lambda x: x**2)), print('-'*30)
    


```python
# FutureWarning: DataFrame.applymap has been deprecated. Use DataFrame.map instead.

# 최근에는 map()함수가 applymap()함수의 기능을 겸함

df1 = pd.DataFrame([[1, 2.12], [3.356, 4.567]])
print(df1), print('='*30)

# 데이터프레임 df에 lambda 함수를 적용하거나 제곱 연산을 실행한 예제
print(df1.applymap(lambda x: x**2)), print('='*30)
print(df1.map(lambda x: x**2))
```

           0      1
    0  1.000  2.120
    1  3.356  4.567
    ==============================
               0          1
    0   1.000000   4.494400
    1  11.262736  20.857489
    ==============================
               0          1
    0   1.000000   4.494400
    1  11.262736  20.857489
    

    <ipython-input-20-d2b680f087ea>:9: FutureWarning: DataFrame.applymap has been deprecated. Use DataFrame.map instead.
      print(df1.applymap(lambda x: x**2)), print('='*30)
    


```python
# 3. 데이터 처리

# 데이터 선택: 1. 라벨로 데이터 선택 (.loc)
df1 = pd.DataFrame(np.random.randn(5, 4), columns=list('ABCD'),
index=pd.date_range('20210701', periods=5))
print(df1), print('='*50)

print(df1.loc['20210702':'20210703'])
```

                       A         B         C         D
    2021-07-01 -0.235171 -0.598080  0.339699 -0.695867
    2021-07-02  1.248476  0.884360 -2.005549 -0.483929
    2021-07-03 -0.792522 -0.167767  0.126749  0.107258
    2021-07-04 -0.503368  0.543597  0.711808  0.914394
    2021-07-05  0.801306 -0.351158 -0.358795 -0.207540
    ==================================================
                       A         B         C         D
    2021-07-02  1.248476  0.884360 -2.005549 -0.483929
    2021-07-03 -0.792522 -0.167767  0.126749  0.107258
    


```python
# 3. 데이터 처리

# 데이터 선택: 1. 라벨로 데이터 선택 (.loc)
ser1 = pd.Series(np.random.randn(4), index=list('abcd'))
print(ser1), print('-'*30)

print(ser1.loc['c':]), print('-'*30)  ###

print(ser1.loc['b']), print('-'*30)

# 시리즈 객체에 동적할당
ser1.loc['c':] = 0
print(ser1)
```

    a    0.478186
    b    2.530093
    c   -0.829002
    d    1.428844
    dtype: float64
    ------------------------------
    c   -0.829002
    d    1.428844
    dtype: float64
    ------------------------------
    2.530092500286559
    ------------------------------
    a    0.478186
    b    2.530093
    c    0.000000
    d    0.000000
    dtype: float64
    


```python
# 3. 데이터 처리

# 데이터 선택: 1. 라벨로 데이터 선택 (.loc) ###
df1 = pd.DataFrame(np.random.randn(5, 4), index=list('abcde'), columns=list('ABCD'))
print(df1), print('='*50)

print(df1.loc[['a', 'b', 'd'], :]), print('='*50)
#print(df1.loc[['a', 'b', 'd']]), print('='*50) # 같은 결과를 만듦

print(df1.loc['c':, 'A':'C']), print('='*50)

print(df1.loc['a'])
```

              A         B         C         D
    a -0.192857 -0.470817  1.976665 -1.091990
    b  1.975813 -0.146923  0.260850 -1.144161
    c  0.339173 -0.594188  0.345793 -0.233776
    d -0.375220 -0.697178  0.079878  0.018531
    e  0.033227 -0.499076 -0.216982 -0.315706
    ==================================================
              A         B         C         D
    a -0.192857 -0.470817  1.976665 -1.091990
    b  1.975813 -0.146923  0.260850 -1.144161
    d -0.375220 -0.697178  0.079878  0.018531
    ==================================================
              A         B         C
    c  0.339173 -0.594188  0.345793
    d -0.375220 -0.697178  0.079878
    e  0.033227 -0.499076 -0.216982
    ==================================================
    A   -0.192857
    B   -0.470817
    C    1.976665
    D   -1.091990
    Name: a, dtype: float64
    


```python
# 3. 데이터 처리

# 데이터 선택: 1. 라벨로 데이터 선택 (.loc) ###

# 인덱스가 정렬되어 있지 Series 생성
ser = pd.Series(list('abcde'), index=[0, 3, 2, 5, 4])
print(ser), print('-'*30)

# 인덱스 레이블이 3부터 5사이의 값(5포함)을 갖는 요소들을 선택
print(ser.loc[3:5]), print('-'*30)
print(ser.loc[2:3]), print('-'*30)

#print(ser.loc[1:6]), print('-'*30) # 에러 발생: 라벨 1과 6이이 없어서

# sort_index(): 인덱스 레이블을 오름차순(default)으로 정렬된 새로운 Series를 반환
print(ser.sort_index()), print('-'*30)

# 정렬된 인덱스 중에서 1과 6은 없지만, 슬라이싱은 존재하는 레이블을 기준으로 작동
print(ser.sort_index().loc[1:6]), print('-'*30)

# 인덱스 레이블을 내림차순으로 정렬된 새로운 Series를 반환
print(ser.sort_index(ascending=False)), print('-'*30)
print(ser.sort_index(ascending=False).loc[1:6])
```

    0    a
    3    b
    2    c
    5    d
    4    e
    dtype: object
    ------------------------------
    3    b
    2    c
    5    d
    dtype: object
    ------------------------------
    Series([], dtype: object)
    ------------------------------
    0    a
    2    c
    3    b
    4    e
    5    d
    dtype: object
    ------------------------------
    2    c
    3    b
    4    e
    5    d
    dtype: object
    ------------------------------
    5    d
    4    e
    3    b
    2    c
    0    a
    dtype: object
    ------------------------------
    Series([], dtype: object)
    


```python
# 3. 데이터 처리

# 데이터 선택: 1. 라벨로 데이터 선택 (.loc) ###
df = pd.DataFrame(np.random.randn(3, 4), columns=['D', 'A', 'C', 'B'])
print(df), print('='*50)


# 열 이름을 기준으로 오름차순 정렬
df_sorted_columns_asc = df.sort_index(axis=1)
print("열 이름 기준 오름차순 정렬:\n", df_sorted_columns_asc), print('='*50)

# 열 이름을 기준으로 내림차순 정렬
df_sorted_columns_desc = df.sort_index(axis=1, ascending=False)
print("열 이름 기준 내림차순 정렬:\n", df_sorted_columns_desc)
```

              D         A         C         B
    0  0.094547  0.474441 -0.018056 -0.111837
    1 -2.099567 -0.522914 -0.648402 -0.722722
    2  0.866296  0.295099 -2.109674 -1.611819
    ==================================================
    열 이름 기준 오름차순 정렬:
               A         B         C         D
    0  0.474441 -0.111837 -0.018056  0.094547
    1 -0.522914 -0.722722 -0.648402 -2.099567
    2  0.295099 -1.611819 -2.109674  0.866296
    ==================================================
    열 이름 기준 내림차순 정렬:
               D         C         B         A
    0  0.094547 -0.018056 -0.111837  0.474441
    1 -2.099567 -0.648402 -0.722722 -0.522914
    2  0.866296 -2.109674 -1.611819  0.295099
    


```python
# 3. 데이터 처리

# 데이터 선택: 2. 위치로 데이터 선택 (.iloc)
ser1 = pd.Series(np.random.randn(5), index=list(range(0, 10, 2)))
print(ser1), print('-'*30)

print(ser1.iloc[:3]), print('-'*30)

print(ser1.iloc[3])
```

    0   -2.112487
    2   -2.098365
    4    2.197695
    6   -0.866636
    8   -0.391818
    dtype: float64
    ------------------------------
    0   -2.112487
    2   -2.098365
    4    2.197695
    dtype: float64
    ------------------------------
    -0.8666361064046363
    ------------------------------
    




    (None, None)




```python
# 3. 데이터 처리

# 데이터 선택: 2. 위치로 데이터 선택 (.iloc)
df1 = pd.DataFrame( np.random.randn(5, 4), index=list(range(0, 10, 2)), columns=list(range(0, 8, 2)))
print(df1), print('='*50)

print(df1.iloc[:2]), print('='*50)

print(df1.iloc[1:3, 0:3]), print('='*50)

print(df1.iloc[[0, 2, 3], [1, 3]]), print('='*50)

print(df1.iloc[1])
```

              0         2         4         6
    0 -1.910830  0.842011  0.170975 -1.428571
    2  0.772548  1.039879  0.042004 -1.190900
    4  0.885613 -0.725073 -0.868190  1.029059
    6  0.088641 -0.813369 -1.074061  0.602078
    8 -0.266482  0.508120  0.050744  0.259292
    ==================================================
              0         2         4         6
    0 -1.910830  0.842011  0.170975 -1.428571
    2  0.772548  1.039879  0.042004 -1.190900
    ==================================================
              0         2         4
    2  0.772548  1.039879  0.042004
    4  0.885613 -0.725073 -0.868190
    ==================================================
              2         6
    0  0.842011 -1.428571
    4 -0.725073  1.029059
    6 -0.813369  0.602078
    ==================================================
    0    0.772548
    2    1.039879
    4    0.042004
    6   -1.190900
    Name: 2, dtype: float64
    


```python
# 3. 데이터 처리

# 데이터 선택: 3. 호출로 데이터 선택
df1 = pd.DataFrame(np.random.randn(5, 4), index=list('abcde'), columns=list('ABCD'))
print(df1), print('='*50)

print(df1.loc[lambda df: df.A>0, :]), print('='*50)
#print(df1.loc[lambda df: df.A>0]), print('='*50) # 위와 같은 결과

print(df1.loc[:, lambda df: ['A', 'B']]), print('='*50)

print(df1.iloc[:, lambda df: [0, 1]]), print('='*50)

print(df1[lambda df: df.columns[0]]), print('='*50)

print(df1.A.loc[lambda ser: ser > 0])
```

              A         B         C         D
    a -1.058404 -0.432219 -0.430031 -0.800542
    b  1.397816 -0.208975  0.146818  0.598320
    c  1.435913 -0.549842  1.392211 -0.463083
    d  0.294567 -0.639996 -0.908162  1.204938
    e  0.262261  1.057673  1.010982  0.287572
    ==================================================
              A         B         C         D
    b  1.397816 -0.208975  0.146818  0.598320
    c  1.435913 -0.549842  1.392211 -0.463083
    d  0.294567 -0.639996 -0.908162  1.204938
    e  0.262261  1.057673  1.010982  0.287572
    ==================================================
              A         B
    a -1.058404 -0.432219
    b  1.397816 -0.208975
    c  1.435913 -0.549842
    d  0.294567 -0.639996
    e  0.262261  1.057673
    ==================================================
              A         B
    a -1.058404 -0.432219
    b  1.397816 -0.208975
    c  1.435913 -0.549842
    d  0.294567 -0.639996
    e  0.262261  1.057673
    ==================================================
    a   -1.058404
    b    1.397816
    c    1.435913
    d    0.294567
    e    0.262261
    Name: A, dtype: float64
    ==================================================
    b    1.397816
    c    1.435913
    d    0.294567
    e    0.262261
    Name: A, dtype: float64
    


```python
# 3. 데이터 처리

# 데이터 설정과 검색: 1. 데이터 확장 및 변경
ser = pd.Series(np.arange(3))
print(ser), print('='*20)

ser[5] = 7
print(ser), print('='*20)


df = pd.DataFrame(np.arange(9).reshape(3, 3), columns=['A', 'B', 'C'])
print(df), print('='*20)

df.loc[:, 'D'] = df.loc[:, 'A']
print(df), print('='*20)

df.loc[3] = 7
print(df), print('='*20)

# at, iat 메소드
print(ser.iat[3]), print('='*20)
print(ser.at[5]), print('='*20)
df.at[3, 'E'] = 7
df.iat[3, 0] = 2
print(df)
```

    0    0
    1    1
    2    2
    dtype: int64
    ====================
    0    0
    1    1
    2    2
    5    7
    dtype: int64
    ====================
       A  B  C
    0  0  1  2
    1  3  4  5
    2  6  7  8
    ====================
       A  B  C  D
    0  0  1  2  0
    1  3  4  5  3
    2  6  7  8  6
    ====================
       A  B  C  D
    0  0  1  2  0
    1  3  4  5  3
    2  6  7  8  6
    3  7  7  7  7
    ====================
    7
    ====================
    7
    ====================
       A  B  C  D    E
    0  0  1  2  0  NaN
    1  3  4  5  3  NaN
    2  6  7  8  6  NaN
    3  2  7  7  7  7.0
    


```python
# 3. 데이터 처리

# 데이터 설정과 검색: 2. 불리언 벡터로 데이터 필터링
ser = pd.Series(range(-3, 3))
print(ser), print('='*20)

print(ser[ser > 0]), print('='*20)

print(ser[(ser < -1) | (ser > 1)]), print('='*20)

print(ser[~(ser < 2)])
```

    0   -3
    1   -2
    2   -1
    3    0
    4    1
    5    2
    dtype: int64
    ====================
    4    1
    5    2
    dtype: int64
    ====================
    0   -3
    1   -2
    5    2
    dtype: int64
    ====================
    5    2
    dtype: int64
    


```python
# 3. 데이터 처리

# 데이터 설정과 검색: 2. 불리언 벡터로 데이터 필터링
ser = pd.Series(range(-3, 3))
print(ser), print('='*20)

# isin 메소드
print(ser[::-1].isin([-3, -1, 2])), print('='*20)

print(ser[ser[::-1].isin([-3, -1, 2])]), print('='*20)

print(ser.index.isin([2, 4, 6])), print('='*20)

print(ser[ser.index.isin([2, 4, 6])]), print('='*40)


df = pd.DataFrame(np.arange(9).reshape(3, 3), columns=['A', 'B', 'C'])
print(df), print('='*20)

print(df[df['A'] < 3])
```

    0   -3
    1   -2
    2   -1
    3    0
    4    1
    5    2
    dtype: int64
    ====================
    5     True
    4    False
    3    False
    2     True
    1    False
    0     True
    dtype: bool
    ====================
    0   -3
    2   -1
    5    2
    dtype: int64
    ====================
    [False False  True False  True False]
    ====================
    2   -1
    4    1
    dtype: int64
    ========================================
       A  B  C
    0  0  1  2
    1  3  4  5
    2  6  7  8
    ====================
       A  B  C
    0  0  1  2
    


```python
# 3. 데이터 처리

# 데이터 설정과 검색: 2. 불리언 벡터로 데이터 필터링
df = pd.DataFrame({'no': [1, 2, 3], 'ha': ['a', 'b', 'c'], 'hi': ['m', 'n', 'o']})
print(df), print('='*50)

val = ['a', 'n', 1, 3]
print(df.isin(val)), print('='*50)

val = {'ha': ['a', 'c'], 'no': [1, 2]}
print(df.isin(val)), print('='*50)

val = {'ha': ['a', 'c'], 'hi': ['m', 'o'], 'no': [1, 2]}
print(df.isin(val)), print('='*50)

mask = df.isin(val).all(1)  # 모두가 True인 것만 True
print(df[mask])
```

       no ha hi
    0   1  a  m
    1   2  b  n
    2   3  c  o
    ==================================================
          no     ha     hi
    0   True   True  False
    1  False  False   True
    2   True  False  False
    ==================================================
          no     ha     hi
    0   True   True  False
    1   True  False  False
    2  False   True  False
    ==================================================
          no     ha     hi
    0   True   True   True
    1   True  False  False
    2  False   True   True
    ==================================================
       no ha hi
    0   1  a  m
    


```python
# 3. 데이터 처리

# 데이터 설정과 검색: 3. take 메소드로 검색

index = pd.Index(np.random.randint(0, 1000, 6))
print(index), print('='*50)

positions = [0, 2, 5]
print(index[positions]), print('='*50)

print(index.take(positions)), print('='*50)


ser = pd.Series(np.random.randn(10))
print(ser.iloc[positions]), print('='*50)

print(ser.take(positions)), print('='*50)


df = pd.DataFrame(np.random.randn(5, 3))
print(df.take([1, 4, 3])), print('='*50)

print(df.take([0, 2], axis=1))

```

    Index([628, 131, 102, 173, 713, 723], dtype='int64')
    ==================================================
    Index([628, 102, 723], dtype='int64')
    ==================================================
    Index([628, 102, 723], dtype='int64')
    ==================================================
    0   -1.198533
    2    0.022278
    5   -1.002791
    dtype: float64
    ==================================================
    0   -1.198533
    2    0.022278
    5   -1.002791
    dtype: float64
    ==================================================
              0         1         2
    1 -0.962585 -0.072540 -1.994170
    4 -0.961239 -1.734835  1.826933
    3  0.208758 -0.393414  1.361795
    ==================================================
              0         2
    0 -0.046307  0.627359
    1 -0.962585 -1.994170
    2 -0.121522  0.635596
    3  0.208758  1.361795
    4 -0.961239  1.826933
    


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
# DataFrame간 산술 연산은 인덱스와 컬럼 레이블을 정렬 후, 대응하는 라벨에 대해 연산
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

# 'pad'는 'ffill'과 같이 바로 이전의 유효한 값으로 채움
print(df2.fillna(method='pad')), print('='*50)  # FutureWarning

df2.loc['c', 'three'] = np.nan
print(df2), print('='*50)

print(df2.mean()), print('='*50)

print(df2.fillna(df2.mean()))
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
    ==================================================
    one      2.0
    two      5.0
    three    2.0
    dtype: float64
    ==================================================
       one  two  three
    a  1.0  4.0    2.0
    b  2.0  5.0    2.0
    c  3.0  6.0    2.0
    d  2.0  5.0    2.0
    

    <ipython-input-28-759e6e7260ca>:15: FutureWarning: DataFrame.fillna with 'method' is deprecated and will raise in a future version. Use obj.ffill() or obj.bfill() instead.
      print(df2.fillna(method='pad')), print('='*50)
    


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

df2.iloc[1:2, 1:2] = 2.0
print(df2), print('='*50)

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
    b  2.0  2.0    2.0
    c  3.0  6.0    2.0
    d  NaN  NaN    2.0
    ==================================================
       one  two  three
    a  1.0  4.0    2.0
    b  2.0  2.0    2.0
    c  3.0  6.0    2.0
    ==================================================
       three
    a    2.0
    b    2.0
    c    2.0
    d    2.0
    ==================================================
    a    4.0
    b    2.0
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

print(df.dropna()), print('='*50)
print(df.dropna(axis='columns')), print('='*50)

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
print(multi_index_arrays)

```

    MultiIndex([(  'IVE', 'Wonyoung'),
                (  'IVE',      'Liz'),
                ('AESPA',   'Winter'),
                ('AESPA',   'Karina')],
               names=['그룹', '이름'])
    


```python
# 3. 데이터 처리

# 멀티인덱스: 객체 생성 (2) from_frame
df = pd.DataFrame([['ha', 'one'], ['ha', 'two'], ['ho', 'one'], ['ho', 'two']],columns=['1st', '2nd'])
pd.MultiIndex.from_frame(df)
```




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

# 멀티인덱스:배열 리스트를 직접 시리즈나 데이터프레임에 입력하여 멀티인덱스를 자동으로 생성
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

    [array(['ha', 'ha', 'hi', 'hi', 'ho', 'ho'], dtype='<U2'), array(['one', 'two', 'one', 'two', 'one', 'two'], dtype='<U3')]
    ==================================================
    ha  one    1.635420
        two   -1.417831
    hi  one   -1.042384
        two   -1.748646
    ho  one   -0.055148
        two   -0.600543
    dtype: float64
    ==================================================
                   0         1         2
    ha one  0.857689 -1.503627 -0.681263
       two -1.006823  0.708220 -1.831113
    hi one -0.254959  0.584394  0.182880
       two  0.613934 -0.154850  0.778190
    ho one  1.326810  1.410717 -0.978157
       two -0.783925 -1.109910 -0.344919
    ==================================================
             ha                  hi                  ho          
            one       two       one       two       one       two
    A -0.695038  1.377937 -1.420908 -1.133828  1.615536 -0.843474
    B  1.199036  1.077299  0.838108  0.960821 -1.732538 -0.377530
    C -0.235057 -0.165980 -2.929994  0.572567 -0.412517 -0.293438
    


```python
# 3. 데이터 처리

# 멀티인덱스: 인덱싱 (1)
df = pd.DataFrame(np.random.randn(3, 6), index=['A', 'B', 'C'], columns=arr)
print(df), print('='*50)

print(df['ha']), print('='*50)

print(df['ha']['one']), print('='*50)


ser = pd.Series(np.random.randn(6), index=arr)
print(ser), print('='*50)

print(ser.reindex([('ho', 'one'), ('ha', 'two')]))
```

             ha                  hi                  ho          
            one       two       one       two       one       two
    A -0.636044  0.155594 -1.114541  1.179070  0.344082  0.987630
    B  0.484571 -0.452022 -0.451587 -2.487041 -0.287674 -0.410189
    C -1.636216  1.071907  1.191265  0.880473  0.331345  0.050189
    ==================================================
            one       two
    A -0.636044  0.155594
    B  0.484571 -0.452022
    C -1.636216  1.071907
    ==================================================
    A   -0.636044
    B    0.484571
    C   -1.636216
    Name: one, dtype: float64
    ==================================================
    ha  one    0.417842
        two    0.552320
    hi  one   -0.714970
        two    0.106901
    ho  one    1.476040
        two    0.743375
    dtype: float64
    ==================================================
    ho  one    1.47604
    ha  two    0.55232
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

# 멀티인덱스: 순서정렬
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
print(df.T.sort_index(level=1, axis=1))
```

    [('ho', 'two'), ('ho', 'one'), ('hi', 'one'), ('ha', 'two'), ('ha', 'one'), ('hi', 'two')]
    ==================================================
    [('hi', 'one'), ('ha', 'one'), ('ha', 'two'), ('ho', 'one'), ('ho', 'two'), ('hi', 'two')]
    ==================================================
    hi  one   -0.025014
    ha  one   -2.114292
        two    0.082206
    ho  one    0.954045
        two   -0.037451
    hi  two    0.726539
    dtype: float64
    ==================================================
    ha  one   -2.114292
        two    0.082206
    hi  one   -0.025014
        two    0.726539
    ho  one    0.954045
        two   -0.037451
    dtype: float64
    ==================================================
    ha  one   -2.114292
        two    0.082206
    hi  one   -0.025014
        two    0.726539
    ho  one    0.954045
        two   -0.037451
    dtype: float64
    ==================================================
    ha  one   -2.114292
    hi  one   -0.025014
    ho  one    0.954045
    ha  two    0.082206
    hi  two    0.726539
    ho  two   -0.037451
    dtype: float64
    ==================================================
    1st  2nd
    ha   one   -2.114292
         two    0.082206
    hi   one   -0.025014
         two    0.726539
    ho   one    0.954045
         two   -0.037451
    dtype: float64
    ==================================================
    1st  2nd
    ha   one   -2.114292
    hi   one   -0.025014
    ho   one    0.954045
    ha   two    0.082206
    hi   two    0.726539
    ho   two   -0.037451
    dtype: float64
    ==================================================
                   A         B         C
    ha one  0.186847  0.195396  0.899658
       two -1.184557 -1.376633  1.289674
    hi one  1.250534  0.677585  0.820775
       two -1.283592 -0.154793  1.070275
    ho one  0.009752 -0.846654  0.103825
       two -0.835926 -0.680803  0.091563
    ==================================================
             ha        hi        ho        ha        hi        ho
            one       one       one       two       two       two
    A  0.186847  1.250534  0.009752 -1.184557 -1.283592 -0.835926
    B  0.195396  0.677585 -0.846654 -1.376633 -0.154793 -0.680803
    C  0.899658  0.820775  0.103825  1.289674  1.070275  0.091563
    
