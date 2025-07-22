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

# DataFrameGroupBy 객체가 생성되었을 뿐, 아직 눈으로 그룹들을 직접 볼 수는 없음
# 이 DataFrameGroupBy 객체는 파이썬의 이터러블(iterable)한 특성을 가지고 있음
# 이터러블 객체는 for 루프와 같이 요소를 하나씩 꺼내서 반복적으로 처리할 수 있음
grouped1 = df.groupby('A')
print(grouped1), print('='*50)

# grouped1을 list로 변환하면 각 그룹의 키와 해당 DataFrame을 볼 수 있음
list_of_groups = list(grouped1)
print(f"list(grouped1)의 첫 번째 요소 (키, DataFrame):\n{list_of_groups[0]}")
print('='*50)

# 딕셔너리로 변환하여 그룹 키를 딕셔너리 키로, 그룹 DataFrame을 딕셔너리 값으로 저장
gr_dict = dict(list_of_groups)
print(f"dict(list_of_groups):\n{gr_dict}"), print('='*50)

print(grouped1.groups)
```

        A    B     Data1     Data2
    0  ha  one -2.248812 -0.835524
    1  hi  two  1.285709 -0.457976
    2  ho  one  0.355776  0.344194
    3  ha  one -0.135937 -0.138679
    4  ho  two -1.489563  1.010608
    ==================================================
    <pandas.core.groupby.generic.DataFrameGroupBy object at 0x000001A923547EC0>
    ==================================================
    list(grouped1)의 첫 번째 요소 (키, DataFrame):
    ('ha',     A    B     Data1     Data2
    0  ha  one -2.248812 -0.835524
    3  ha  one -0.135937 -0.138679)
    ==================================================
    dict(list_of_groups):
    {'ha':     A    B     Data1     Data2
    0  ha  one -2.248812 -0.835524
    3  ha  one -0.135937 -0.138679, 'hi':     A    B     Data1     Data2
    1  hi  two  1.285709 -0.457976, 'ho':     A    B     Data1     Data2
    2  ho  one  0.355776  0.344194
    4  ho  two -1.489563  1.010608}
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
    2  ho  one  0.070154  0.891122
    4  ho  two  0.501409  0.682108
    ==================================================
        A    B     Data1     Data2
    2  ho  one -0.231873 -0.387598
    4  ho  two  0.053033  0.180301
    


```python
# 2. 데이터의 그룹 연산

# 데이터 객체를 그룹 연산: 그룹 객체의 반복처리
df = pd.DataFrame({'A': ['ha', 'hi', 'ho', 'ha', 'ho'],
                   'B': ['one', 'two', 'one', 'one', 'two'],
                   'Data1': np.random.randn(5),
                   'Data2': np.random.randn(5)})
print(df), print('='*50)

grouped1 = df.groupby('A')
print(grouped1), print('='*50)

for name, group in grouped1:
    print(name)

print('='*50)

for name, group in grouped1:
    print(name)
    print(group)
    print('-'*50)
```

        A    B     Data1     Data2
    0  ha  one -0.169027  0.136886
    1  hi  two -0.477366 -0.422816
    2  ho  one -0.068780  0.415388
    3  ha  one  1.020587  0.594460
    4  ho  two  1.377188 -0.334472
    ==================================================
    <pandas.core.groupby.generic.DataFrameGroupBy object at 0x000001A92383A120>
    ==================================================
    ha
    hi
    ho
    ==================================================
    ha
        A    B     Data1     Data2
    0  ha  one -0.169027  0.136886
    3  ha  one  1.020587  0.594460
    --------------------------------------------------
    hi
        A    B     Data1     Data2
    1  hi  two -0.477366 -0.422816
    --------------------------------------------------
    ho
        A    B     Data1     Data2
    2  ho  one -0.068780  0.415388
    4  ho  two  1.377188 -0.334472
    --------------------------------------------------
    


```python
# 2. 데이터의 그룹 연산

# 데이터 객체를 그룹 연산
print(df), print('='*50)

# n1은 첫 번째 그룹화 기준 컬럼('A')의 값을 받음
# n2는 두 번째 그룹화 기준 컬럼('B')의 값을 받음
for (n1, n2), group in df.groupby(['A', 'B']):
    print((n1, n2)), print('-'*20)
    print(group), print('-'*50)
```

        A    B     Data1     Data2
    0  ha  one -0.169027  0.136886
    1  hi  two -0.477366 -0.422816
    2  ho  one -0.068780  0.415388
    3  ha  one  1.020587  0.594460
    4  ho  two  1.377188 -0.334472
    ==================================================
    ('ha', 'one')
    --------------------
        A    B     Data1     Data2
    0  ha  one -0.169027  0.136886
    3  ha  one  1.020587  0.594460
    --------------------------------------------------
    ('hi', 'two')
    --------------------
        A    B     Data1     Data2
    1  hi  two -0.477366 -0.422816
    --------------------------------------------------
    ('ho', 'one')
    --------------------
        A    B    Data1     Data2
    2  ho  one -0.06878  0.415388
    --------------------------------------------------
    ('ho', 'two')
    --------------------
        A    B     Data1     Data2
    4  ho  two  1.377188 -0.334472
    --------------------------------------------------
    


```python
# 2. 데이터의 그룹 연산

# 데이터 객체를 그룹 연산: GroupBy 객체 속성
print(df), print('='*50)

print(df['Data2']), print('='*50)

grouped2 = df['Data2'].groupby(df['A'])
print(grouped2), print('='*50)
print(grouped2.mean()), print('='*50)

grouped3 = df['Data1'].groupby([df['A'], df['B']])
print(grouped3.groups), print('='*50)
print(grouped3.mean())
```

        A    B     Data1     Data2
    0  ha  one -0.169027  0.136886
    1  hi  two -0.477366 -0.422816
    2  ho  one -0.068780  0.415388
    3  ha  one  1.020587  0.594460
    4  ho  two  1.377188 -0.334472
    ==================================================
    0    0.136886
    1   -0.422816
    2    0.415388
    3    0.594460
    4   -0.334472
    Name: Data2, dtype: float64
    ==================================================
    <pandas.core.groupby.generic.SeriesGroupBy object at 0x000001A923B59910>
    ==================================================
    A
    ha    0.365673
    hi   -0.422816
    ho    0.040458
    Name: Data2, dtype: float64
    ==================================================
    {('ha', 'one'): [0, 3], ('hi', 'two'): [1], ('ho', 'one'): [2], ('ho', 'two'): [4]}
    ==================================================
    A   B  
    ha  one    0.425780
    hi  two   -0.477366
    ho  one   -0.068780
        two    1.377188
    Name: Data1, dtype: float64
    


```python
# 2. 데이터의 그룹 연산

# 데이터 객체를 그룹 연산: GroupBy 객체 속성

# df와 길이가 같은 배열로 이루어진 시리즈 객체와 리스트를
# groupby( )에 전달하여 연산을 실행할 수도 있음
material = np.array(['water', 'oil', 'oil' ,'water', 'oil'])
time = ['1hr', '1hr', '2hr', '2hr', '1hr']

print(df), print('='*50)
print(material), print('='*50)
print(time), print('='*50)
print(df['Data1'].groupby([material, time]).mean())

# df_aug = df.copy()
# print(df_aug), print('='*50)
# df_aug['Material'] = material
# df_aug['Time'] = time
# print(df_aug), print('='*50)
# print(df_aug['Data1'].groupby([material, time]).mean())
```

        A    B     Data1     Data2
    0  ha  one -0.169027  0.136886
    1  hi  two -0.477366 -0.422816
    2  ho  one -0.068780  0.415388
    3  ha  one  1.020587  0.594460
    4  ho  two  1.377188 -0.334472
    ==================================================
    ['water' 'oil' 'oil' 'water' 'oil']
    ==================================================
    ['1hr', '1hr', '2hr', '2hr', '1hr']
    ==================================================
    oil    1hr    0.449911
           2hr   -0.068780
    water  1hr   -0.169027
           2hr    1.020587
    Name: Data1, dtype: float64
    


```python
# 2. 데이터의 그룹 연산

# 데이터 객체를 그룹 연산: GroupBy 순서 정렬
df2 = pd.DataFrame({'A': ['ho', 'hi', 'ha', 'ha'],
                    'B': ['two', 'one', 'two', 'two'],
                    'Data1': np.random.randn(4)})

print(df2), print('='*50)
print(df2.groupby('A').sum()), print('='*50)
print(df2.groupby('A', sort=False).sum())

# mean()을 사용할 때는 주의
#print(df2.groupby(['A']).mean())
```

        A    B     Data1
    0  ho  two  0.490391
    1  hi  one -0.135906
    2  ha  two -0.677599
    3  ha  two  0.908525
    ==================================================
             B     Data1
    A                   
    ha  twotwo  0.230926
    hi     one -0.135906
    ho     two  0.490391
    ==================================================
             B     Data1
    A                   
    ho     two  0.490391
    hi     one -0.135906
    ha  twotwo  0.230926
    


```python
# 2. 데이터의 그룹 연산

# 데이터 객체를 그룹 연산: 멀티인덱스가 있는 개체를 그룹 연산
arr = [['ha', 'ha', 'hi', 'hi', 'ho', 'ho'], ['one', 'two', 'one', 'one', 'two', 'two']]
ind = pd.MultiIndex.from_arrays(arr, names=['1st', '2nd'])

ser = pd.Series(np.random.randn(6), index=ind)
print(ser), print('='*50)
print(ser.index), print('='*50)

grouped = ser.groupby(level=0)
print(grouped.mean()), print('='*50)

print(ser.groupby(level=1).mean()), print('='*50)
print(ser.groupby(level='2nd').mean())
```

    1st  2nd
    ha   one   -0.749425
         two    0.175374
    hi   one   -0.015161
         one   -1.496129
    ho   two    0.668025
         two   -0.764227
    dtype: float64
    ==================================================
    MultiIndex([('ha', 'one'),
                ('ha', 'two'),
                ('hi', 'one'),
                ('hi', 'one'),
                ('ho', 'two'),
                ('ho', 'two')],
               names=['1st', '2nd'])
    ==================================================
    1st
    ha   -0.287026
    hi   -0.755645
    ho   -0.048101
    dtype: float64
    ==================================================
    2nd
    one   -0.753572
    two    0.026390
    dtype: float64
    ==================================================
    2nd
    one   -0.753572
    two    0.026390
    dtype: float64
    


```python
# 2. 데이터의 그룹 연산

# GroupBy 객체를 그룹별 연산 및 변환: 데이터 집계하기
df = pd.DataFrame({'A': ['ha', 'hi', 'ho', 'ha', 'ho'],
                   'B': ['one', 'two', 'one', 'one', 'two'],
                   'Data1': np.random.randn(5),
                   'Data2': np.random.randn(5)})

print(df), print('='*50)

grouped1 = df.groupby('A')
print(grouped1.agg(np.sum)), print('='*50)
#print(grouped1.agg('sum')), print('='*50) # 위와 같은 결과

grouped2 = df.groupby(['A', 'B'])
print(f'grouped.size(): \n{grouped2.size()}'), print('='*50)
print(grouped2.agg('sum')), print('='*50)

# as_index=False: 그룹 키가 일반 컬럼으로 유지되고, 새로운 정수 인덱스가 생성
grouped3 = df.groupby(['A', 'B'], as_index=False)
print(grouped3.agg(np.sum)) # grouped2.agg('sum')과 결과는 같지만, 인덱스가 다름
```

        A    B     Data1     Data2
    0  ha  one  0.424156 -0.038641
    1  hi  two  1.025926  0.236049
    2  ho  one  0.055039  0.168925
    3  ha  one  0.319258 -0.517434
    4  ho  two  0.776229 -0.348690
    ==================================================
             B     Data1     Data2
    A                             
    ha  oneone  0.743415 -0.556076
    hi     two  1.025926  0.236049
    ho  onetwo  0.831268 -0.179765
    ==================================================
    grouped.size(): 
    A   B  
    ha  one    2
    hi  two    1
    ho  one    1
        two    1
    dtype: int64
    ==================================================
               Data1     Data2
    A  B                      
    ha one  0.743415 -0.556076
    hi two  1.025926  0.236049
    ho one  0.055039  0.168925
       two  0.776229 -0.348690
    ==================================================
        A    B     Data1     Data2
    0  ha  one  0.743415 -0.556076
    1  hi  two  1.025926  0.236049
    2  ho  one  0.055039  0.168925
    3  ho  two  0.776229 -0.348690
    

    C:\Users\User\AppData\Local\Temp\ipykernel_17168\1918079573.py:12: FutureWarning: The provided callable <function sum at 0x000001A921195620> is currently using DataFrameGroupBy.sum. In a future version of pandas, the provided callable will be used directly. To keep current behavior pass the string "sum" instead.
      print(grouped1.agg(np.sum)), print('='*50)
    C:\Users\User\AppData\Local\Temp\ipykernel_17168\1918079573.py:21: FutureWarning: The provided callable <function sum at 0x000001A921195620> is currently using DataFrameGroupBy.sum. In a future version of pandas, the provided callable will be used directly. To keep current behavior pass the string "sum" instead.
      print(grouped3.agg(np.sum)) # grouped2.agg('sum')과 결과는 같지만, 인덱스가 다름
    


```python
# 2. 데이터의 그룹 연산

# GroupBy 객체를 그룹별 연산 및 변환: 한 번에 여러 함수 적용하기
df = pd.DataFrame({'A': ['ha', 'hi', 'ho', 'ha', 'ho'],
                   'Data1': np.random.randn(5),
                   'Data2': np.random.randn(5)})
print(df), print('='*50)

grouped = df.groupby('A')
result = grouped['Data1'].agg(['sum', 'mean', 'std'])
#result = df.groupby('A')['Data1'].agg([np.sum, np.mean, np.std]) # 위 코드와 같음
print(result), print('='*50)

# 함수 자체가 열의 이름이 되는데, 이름을 변경하고 싶으면 rename()메소드를 실행
result2 = grouped['Data1'].agg(['sum','mean']).rename(columns={'sum':'합계', 'mean':'평균'})
print(result2), print('='*50)

# 집계 연산으로 처리된 결과는 멀티인덱스를 가짐
result3 = grouped.agg(['sum', 'mean'])
print(result3), print('='*50)

# rename()메소드를 실행
result4 = grouped.agg(['sum', 'mean']).rename(columns={'sum': '합계','mean': '평균'})
print(result4)
```

        A     Data1     Data2
    0  ha  0.715189 -0.203570
    1  hi  0.144084 -0.706190
    2  ho -1.569879  0.812396
    3  ha  0.107967  0.344885
    4  ho -0.623013  0.173615
    ==================================================
             sum      mean       std
    A                               
    ha  0.823156  0.411578  0.429371
    hi  0.144084  0.144084       NaN
    ho -2.192893 -1.096446  0.669535
    ==================================================
              합계        평균
    A                     
    ha  0.823156  0.411578
    hi  0.144084  0.144084
    ho -2.192893 -1.096446
    ==================================================
           Data1               Data2          
             sum      mean       sum      mean
    A                                         
    ha  0.823156  0.411578  0.141315  0.070658
    hi  0.144084  0.144084 -0.706190 -0.706190
    ho -2.192893 -1.096446  0.986011  0.493005
    ==================================================
           Data1               Data2          
              합계        평균        합계        평균
    A                                         
    ha  0.823156  0.411578  0.141315  0.070658
    hi  0.144084  0.144084 -0.706190 -0.706190
    ho -2.192893 -1.096446  0.986011  0.493005
    


```python
# 2. 데이터의 그룹 연산

# GroupBy 객체를 그룹별 연산 및 변환: 데이터프레임 열들에 각각 다른 함수 적용하기
df = pd.DataFrame({'A': ['ha', 'hi', 'ho', 'ha', 'ho'],
                   'Data1': np.random.randn(5),
                   'Data2': np.random.randn(5)})

print(df), print('='*50)

grouped = df.groupby('A')
result = grouped.agg({'Data1': 'mean', 'Data2': lambda x: np.sum(x)})
print(result), print('='*50)


from collections import OrderedDict

# agg() 함수에 딕셔너리를 전달하면 열의 순서를 임의로 출력함
result1 = grouped.agg({'Data1': 'sum', 'Data2': 'mean'})
print(result1), print('='*50)

# OrderedDict() 메소드로 열을 특정 순서로 정렬함
result2 = grouped.agg(OrderedDict([('Data2', 'mean'), ('Data1', 'sum')]))
print(result2), print('='*50)

# 인덱스 'ha'를 'gold', 'hi'를 'silver', 'ho'를 'gold'로 변경 후, 각각 합계구함
ind = ['gold', 'silver', 'gold']
print(result2.groupby(ind).sum())
```

        A     Data1     Data2
    0  ha -0.651225 -1.391641
    1  hi  0.578765  0.751849
    2  ho -2.113492 -2.257368
    3  ha -0.096680 -0.264105
    4  ho  1.167026  1.027779
    ==================================================
           Data1     Data2
    A                     
    ha -0.373952 -1.655746
    hi  0.578765  0.751849
    ho -0.473233 -1.229590
    ==================================================
           Data1     Data2
    A                     
    ha -0.747905 -0.827873
    hi  0.578765  0.751849
    ho -0.946466 -0.614795
    ==================================================
           Data2     Data1
    A                     
    ha -0.827873 -0.747905
    hi  0.751849  0.578765
    ho -0.614795 -0.946466
    ==================================================
               Data2     Data1
    gold   -1.442668 -1.694371
    silver  0.751849  0.578765
    


```python
# 2. 데이터의 그룹 연산

# GroupBy 객체를 그룹별 연산 및 변환: 자동차 판매 대리점별 영업 현황 데이터 연산과 변환
df = pd.read_excel('car_sales.xlsx')
print(df), print('='*50)

# Branch로 그룹화하고 Ext Price 열을 기준으로 그룹별로 sum
result1 = df.groupby('Branch')['Ext Price'].agg('sum')
print(result1), print('='*50)

# result11 = df.groupby('Branch')['Ext Price'].apply('sum')
# print(result11), print('='*50)

# rename 메소드로 컬럼 이름 변경
result2 = df.groupby('Branch')['Ext Price'].agg('sum').rename('Br_Total')
print(result2), print('='*50)

# 인덱스 초기화
br_total = df.groupby('Branch')['Ext Price'].agg('sum').rename('Br_Total').reset_index()
print(br_total), print('='*50)

df_m = df.merge(br_total)
print(df_m)
```

        Branch  Car Name  Quantity  Unit Price  Ext Price
    0  Yeonnam  Grandeur         7          35        245
    1  Yeonnam    Sonata        11          20        220
    2  Yeonnam    Avante         3          15         45
    3  Sungsan  Grandeur         5          36        180
    4  Sungsan    Sonata        19          19        361
    5  Sungsan    Avante         9          14        126
    6   Yeonhi  Grandeur        10          34        340
    7   Yeonhi    Sonata        13          19        247
    8   Yeonhi    Avante        15          13        195
    ==================================================
    Branch
    Sungsan    667
    Yeonhi     782
    Yeonnam    510
    Name: Ext Price, dtype: int64
    ==================================================
    Branch
    Sungsan    667
    Yeonhi     782
    Yeonnam    510
    Name: Br_Total, dtype: int64
    ==================================================
        Branch  Br_Total
    0  Sungsan       667
    1   Yeonhi       782
    2  Yeonnam       510
    ==================================================
        Branch  Car Name  Quantity  Unit Price  Ext Price  Br_Total
    0  Yeonnam  Grandeur         7          35        245       510
    1  Yeonnam    Sonata        11          20        220       510
    2  Yeonnam    Avante         3          15         45       510
    3  Sungsan  Grandeur         5          36        180       667
    4  Sungsan    Sonata        19          19        361       667
    5  Sungsan    Avante         9          14        126       667
    6   Yeonhi  Grandeur        10          34        340       782
    7   Yeonhi    Sonata        13          19        247       782
    8   Yeonhi    Avante        15          13        195       782
    


```python
# 2. 데이터의 그룹 연산

# GroupBy 객체를 그룹별 연산 및 변환: 자동차 판매 대리점별 영업 현황 데이터 연산과 변환
print(df_m), print('='*50)

# 각 대리점의 차종별 매출액 비율 계산
df_m['Br_Pct'] = df_m['Ext Price'] / df_m['Br_Total']
print(df_m), print('='*50)

### transform(), apply(), agg() 결과 비교
# Branch로 그룹화하고 Ext Price 열을 기준으로 그룹별로 sum
result1 = df.groupby('Branch')['Ext Price'].agg('sum')
print(result1), print('='*50)

result2 = df.groupby('Branch')['Ext Price'].apply('sum')
print(result2), print('='*50)

# transform() 메소드를 이용해 shape이 같은 데이터 반환
result3 = df.groupby('Branch')['Ext Price'].transform('sum')
print(result3), print('='*50)

df['Br_Total'] = df.groupby('Branch')['Ext Price'].transform('sum')
print(df), print('='*50)

df['Br_Pct'] = df['Ext Price'] / df['Br_Total']
print(df)
```

        Branch  Car Name  Quantity  Unit Price  Ext Price  Br_Total
    0  Yeonnam  Grandeur         7          35        245       510
    1  Yeonnam    Sonata        11          20        220       510
    2  Yeonnam    Avante         3          15         45       510
    3  Sungsan  Grandeur         5          36        180       667
    4  Sungsan    Sonata        19          19        361       667
    5  Sungsan    Avante         9          14        126       667
    6   Yeonhi  Grandeur        10          34        340       782
    7   Yeonhi    Sonata        13          19        247       782
    8   Yeonhi    Avante        15          13        195       782
    ==================================================
        Branch  Car Name  Quantity  Unit Price  Ext Price  Br_Total    Br_Pct
    0  Yeonnam  Grandeur         7          35        245       510  0.480392
    1  Yeonnam    Sonata        11          20        220       510  0.431373
    2  Yeonnam    Avante         3          15         45       510  0.088235
    3  Sungsan  Grandeur         5          36        180       667  0.269865
    4  Sungsan    Sonata        19          19        361       667  0.541229
    5  Sungsan    Avante         9          14        126       667  0.188906
    6   Yeonhi  Grandeur        10          34        340       782  0.434783
    7   Yeonhi    Sonata        13          19        247       782  0.315857
    8   Yeonhi    Avante        15          13        195       782  0.249361
    ==================================================
    Branch
    Sungsan    667
    Yeonhi     782
    Yeonnam    510
    Name: Ext Price, dtype: int64
    ==================================================
    Branch
    Sungsan    667
    Yeonhi     782
    Yeonnam    510
    Name: Ext Price, dtype: int64
    ==================================================
    0    510
    1    510
    2    510
    3    667
    4    667
    5    667
    6    782
    7    782
    8    782
    Name: Ext Price, dtype: int64
    ==================================================
        Branch  Car Name  Quantity  Unit Price  Ext Price  Br_Total
    0  Yeonnam  Grandeur         7          35        245       510
    1  Yeonnam    Sonata        11          20        220       510
    2  Yeonnam    Avante         3          15         45       510
    3  Sungsan  Grandeur         5          36        180       667
    4  Sungsan    Sonata        19          19        361       667
    5  Sungsan    Avante         9          14        126       667
    6   Yeonhi  Grandeur        10          34        340       782
    7   Yeonhi    Sonata        13          19        247       782
    8   Yeonhi    Avante        15          13        195       782
    ==================================================
        Branch  Car Name  Quantity  Unit Price  Ext Price  Br_Total    Br_Pct
    0  Yeonnam  Grandeur         7          35        245       510  0.480392
    1  Yeonnam    Sonata        11          20        220       510  0.431373
    2  Yeonnam    Avante         3          15         45       510  0.088235
    3  Sungsan  Grandeur         5          36        180       667  0.269865
    4  Sungsan    Sonata        19          19        361       667  0.541229
    5  Sungsan    Avante         9          14        126       667  0.188906
    6   Yeonhi  Grandeur        10          34        340       782  0.434783
    7   Yeonhi    Sonata        13          19        247       782  0.315857
    8   Yeonhi    Avante        15          13        195       782  0.249361
    


```python
# 2. 데이터의 그룹 연산

# GroupBy 객체를 그룹별 연산 및 변환: 학교별 수학 성적에 분할, 적용, 통합 연산 실행
df = pd.DataFrame({'School': ['Yeonhi', 'Yeonhi', 'Sungsan', 'Sungsan', 'Sungsan'],
                   'Name': ['Haena', 'Gisu', 'Una', 'Naeun', 'Ziho'],
                   'Math_S': [92, 71, 88, 92, 70]})
print(df), print('='*50)

mean_s = df.groupby('School')['Math_S'].agg('mean')
print(mean_s), print('='*50)

print(mean_s.rename('Avg_S')), print('='*50)

avg_score = mean_s.rename('Avg_S').reset_index()
print(avg_score), print('='*50)

# df 객체에 avg_score를 병항
df_new = df.merge(avg_score)
print(df_new), print('='*50)

# apply() 메소드 적용
df['Rating_S'] = df['Math_S'].apply(lambda x: x/100)
print(df)
```

        School   Name  Math_S
    0   Yeonhi  Haena      92
    1   Yeonhi   Gisu      71
    2  Sungsan    Una      88
    3  Sungsan  Naeun      92
    4  Sungsan   Ziho      70
    ==================================================
    School
    Sungsan    83.333333
    Yeonhi     81.500000
    Name: Math_S, dtype: float64
    ==================================================
    School
    Sungsan    83.333333
    Yeonhi     81.500000
    Name: Avg_S, dtype: float64
    ==================================================
        School      Avg_S
    0  Sungsan  83.333333
    1   Yeonhi  81.500000
    ==================================================
        School   Name  Math_S      Avg_S
    0   Yeonhi  Haena      92  81.500000
    1   Yeonhi   Gisu      71  81.500000
    2  Sungsan    Una      88  83.333333
    3  Sungsan  Naeun      92  83.333333
    4  Sungsan   Ziho      70  83.333333
    ==================================================
        School   Name  Math_S  Rating_S
    0   Yeonhi  Haena      92      0.92
    1   Yeonhi   Gisu      71      0.71
    2  Sungsan    Una      88      0.88
    3  Sungsan  Naeun      92      0.92
    4  Sungsan   Ziho      70      0.70
    


```python
# 2. 데이터의 그룹 연산

# GroupBy 객체를 그룹별 연산 및 변환: 학교별 수학 성적에 분할, 적용, 통합 연산 실행
print(df), print('='*50)

math_score = df['Math_S']
grade = []
for x in math_score:
    if x > 90:
        grade.append('A')
    elif x > 80:
        grade.append('B')
    elif x > 70:
        grade.append('C')
    else:
        grade.append('F')

print(grade), print('='*50)

df['Grade'] = grade
print(df)
```

        School   Name  Math_S  Rating_S
    0   Yeonhi  Haena      92      0.92
    1   Yeonhi   Gisu      71      0.71
    2  Sungsan    Una      88      0.88
    3  Sungsan  Naeun      92      0.92
    4  Sungsan   Ziho      70      0.70
    ==================================================
    ['A', 'C', 'B', 'A', 'F']
    ==================================================
        School   Name  Math_S  Rating_S Grade
    0   Yeonhi  Haena      92      0.92     A
    1   Yeonhi   Gisu      71      0.71     C
    2  Sungsan    Una      88      0.88     B
    3  Sungsan  Naeun      92      0.92     A
    4  Sungsan   Ziho      70      0.70     F
    


```python
# 2. 데이터의 그룹 연산

# GroupBy 객체를 그룹별 연산 및 변환: 학교별 수학 성적에 분할, 적용, 통합 연산 실행
print(df), print('='*50)

# transform() 메소드로 학점 연산하는 방법
result = df.groupby('School')['Math_S'].transform('mean')
print(result), print('='*50)

df['Avg_S'] = result
print(df), print('='*50)

df['Above_Avg'] = df['Avg_S'] < df['Math_S']
print(df)


```

        School   Name  Math_S  Rating_S Grade
    0   Yeonhi  Haena      92      0.92     A
    1   Yeonhi   Gisu      70      0.70     F
    2  Sungsan    Una      88      0.88     B
    3  Sungsan  Naeun      92      0.92     A
    4  Sungsan   Ziho      70      0.70     F
    ==================================================
    0    81.000000
    1    81.000000
    2    83.333333
    3    83.333333
    4    83.333333
    Name: Math_S, dtype: float64
    ==================================================
        School   Name  Math_S  Rating_S Grade      Avg_S
    0   Yeonhi  Haena      92      0.92     A  81.000000
    1   Yeonhi   Gisu      70      0.70     F  81.000000
    2  Sungsan    Una      88      0.88     B  83.333333
    3  Sungsan  Naeun      92      0.92     A  83.333333
    4  Sungsan   Ziho      70      0.70     F  83.333333
    ==================================================
        School   Name  Math_S  Rating_S Grade      Avg_S  Above_Avg
    0   Yeonhi  Haena      92      0.92     A  81.000000       True
    1   Yeonhi   Gisu      70      0.70     F  81.000000      False
    2  Sungsan    Una      88      0.88     B  83.333333       True
    3  Sungsan  Naeun      92      0.92     A  83.333333       True
    4  Sungsan   Ziho      70      0.70     F  83.333333      False
    


```python

```
