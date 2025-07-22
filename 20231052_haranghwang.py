# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 12:23:51 2025

@author: haran
"""

# %% Q1. 20231052_haranghwang

import numpy as np
import pandas as pd

df = pd.read_excel('./patients.csv')
df1 = pd.read_excel('./patients.csv', sheet_name='patients_df')
print(df1)

df2 = pd.read_excel('./health_checkups.csv')
df3 = pd.read_excel('./health_checkups.csv', sheet_name='health_checkups_df')
print(df3)


# %% Q2. 20231052_haranghwang

df4 = np.genfromtxt('./health_checkups.csv', delimiter=',',  dtype=<type>, delimiter=None, skip_header=0, missing_values=None, filling_values=None)

print(df4)



# %% Q3. 20231052_haranghwang



missing_data = np.genfromtxt('./health_checkups.csv', delimiter=',' , missing_values='NA', filling_values=0)
print (missing_data)

df5 = pd.series{'./health_checkups.csv'}
pandas.crosstab(index, columns, values='NaN',  aggfunc='count')

# %% Q4. 20231052_haranghwang


df5 = pd.DataFrame(columns='missing_data') 
df6 = np.mean(df5['missing_data'])
 result1 = df1.combine_first(df6)



# %% Q5. 20231052_haranghwang


df7 = pd.DataFrame('health_checkups_df')
df8 = pd.DataFrame('patients_df')

df9 = pandas.merge(df7, df8, how='left', left_on='Patient ID', left_index=False, 
right_index=False, sort=False)




# %% Q6. 20231052_haranghwang

 import matplotlib.pyplot as plt

plt.figure(figsize=(4,8))

 plt.subplot(2, 1, 1)
 x = 'Cholesterol'
y = 'BloodSugar'
s = [30]
  plt.scatter(x = , y = y, s=s, c='red', marker='o', alpha=1)
 
 plt.subplot(2, 1, 2)
plt.plot(x, y, marker='o')
 plt.xticks(ticks=[5, 10, 15, 20], fontsize=10, color='darkgreen')
 plt.yticks(ticks=[5, 10, 15, 20], fontsize=10, color='darkgreen')
 plt.xlabel("Cholesterol"), plt.ylabel("BloodSugar"), plt.title("CheckupData")
 plt.grid(True, linestyle='--', alpha=0.7)
 plt.tight_layout()
 plt.show()





# %% Q7. 20231052_haranghwang



missing_data = np.genfromtxt('merged_health_df', delimiter=',' , missing_values='NA', filling_values=0)
print (missing_data)

df10 = pd.series{'merged_health_df'}
d11 = pandas.crosstab(index, columns, values='NaN',  aggfunc='count')
print(d11)

# %% Q8. 20231052_haranghwang


df = merged_health_df
print(df.del.loc[200<df['Height']]&df.loc[df['Height']<140])
print(df.del.loc[150<df['Weight']]&df.loc[df['Weight']<40])



# %% Q9. 20231052_haranghwang

df = merged_health_df.pandas.crosstab(index, columns, values='Index', aggfunc='count')
print(df)




# %% Q10. 20231052_haranghwang

df1 = merged_health_df.loc['Disease_Databetes']=True
df2 = merged_health_df.loc['Disease_Databetes']=False


print(df1.std)
print(df2.std)

print(df1.apply(np.mean))
print(df2.apply(np.mean))




# %% Q11. 20231052_haranghwang


RiskScore = pd.concat(frames)

print(result1.loc['RiskScore'])



# %% Q12. 20231052_haranghwang


df_sorted_columns_desc = df.sort_index(axis=1, ascending=False)

 print("열이름기준내림차순정렬:\n", df_sorted_columns_desc)





# %% Q13. 20231052_haranghwang



age_category = df.groupby('Category')

print(df.loc [20 <= df['Age'] < 30])
print(df.loc [30 <= df['Age'] < 45])
print(df.loc [45 <= df['Age'] < 60])
print(df.loc [df['Age'] >= 60])



# %% Q14. 20231052_haranghwang


df['AgeCategory'] = df.loc['merged_health_df']






# %% Q15. 20231052_황하랑


multi_index_arrays = pd.MultiIndex.from_arrays(arrays, names=['City','AgeCategory'])

dfp = df.pivot_tabl.mean(values='health_checkups_df', columns=['SystolicBP', 'DiastolicBP', 'BloodSugar', 'Cholesterol'])

print (dfp)



# %% Q16. 20231052_haranghwang


DataFrame.to_csv('final_health_data.csv', sep=',', na_rep='NaN', header=True, index=False, mode='w')




