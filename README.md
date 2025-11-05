Assign 1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('Heart.csv')
print(df)

df.shape
df.head()
df.tail()
df.dtypes
df.isnull().sum()
(df==0).sum()
df['Age'].mean()
df.columns
from sklearn.model_selection import train_test_split
selected_columns = ['Age', 'Sex', 'ChestPain', 'RestBP', 'Chol']
selected_data = df[selected_columns]
train_data, test_data = train_test_split(selected_data, test_size=0.25, random_state=42)

print("Training Data Shape:", train_data.shape)
print("Testing Data Shape:", test_data.shape)
df['Sex']=df['Sex'].replace([1],'Male')
df['Sex']=df['Sex'].replace([0],'Female')
print(df)


df['Age'].hist(figsize=(10,13))
plt.title('Age Histogram')
