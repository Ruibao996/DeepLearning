import cv2
import numpy as np
import pandas as pd

num = np.random.normal(0, 1, (3, 4))
print(num)
num[num < 0.5] = 0
print(num)
# np.where(condition, x, y) 满足条件(condition)，输出x，不满足输出y。
print(np.where(num < 0.5, 1, 0))

# 使用列表创建Series，索引值为默认值
s1 = pd.Series([1, 1, 1])
print(s1)
# 使用字典创建，索引值为字典的key
s2 = pd.Series({'a': 1, 'b': 2, 'c': 3})
print(s2)
# 使用range()函数生成的迭代序列设置索引值
s3 = pd.Series([3.4, 2.5, 0.8], index=range(3, 6))
print(s3)

# 使用列表创建DataFrame，索引值为默认值
df1 = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(df1)
# 使用字典创建DataFrame，索引值为字典的key
df2 = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
print(df2)
# 由列表组成的字典创建DataFrame，索引值为字典的key
lista = [1, 3, 5, 7, 9]
listb = ['a', 'b', 'c', 'd', 'e']
df3 = pd.DataFrame({'col1': lista, 'col2': listb})
print(df3)

ser = pd.Series(np.arange(4), index=['A', 'B', 'C', 'D'])
data = pd.DataFrame(np.arange(16).reshape(4, 4), index=[
                    'BJ', 'SH', 'SZ', 'GZ'], columns=['q', 'r', 's', 't'])
print(data)
print(data.loc[['SH', 'GZ'], ['r', 's']])
print(data.iloc[:-1, 1:3])

p = 'life can be good, life can be sad, life is mostly cheerful, but sometimes sad.'
pList = p.split()
tmpList = [1]*len(pList)  # 创建一个长度为pList的列表，每个元素为1
dfDict = pd.DataFrame({'word': pList, 'count': tmpList})
print(dfDict)
# 分组汇总，使用reset_index()方法重置索引
dfDict = dfDict.groupby('word').count().reset_index()
print(dfDict)
