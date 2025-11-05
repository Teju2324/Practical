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


Assign 6
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

df = pd.read_csv('C:/Users/HP/Music/pima-indians-diabetes.csv')

df.info()

df.columns

x=df.iloc[:,0:-1].values
y=df.iloc[:,8].values

!pip install ann_visualizer

from ann_visualizer.visualize import ann_viz;

model=Sequential()
model.add (Dense (12, input_dim=8,activation='relu'))
model.add (Dense(8,activation='relu'))
model.add (Dense(1, activation='sigmoid'))
model.compile(loss ='binary_crossentropy',optimizer ='adam',metrics =['accuracy'])
model.fit(x, y, epochs=100, batch_size=10)

_, accuracy = model.evaluate(x, y)
print('Accuracy: %.2f%%' % (accuracy * 100))

from tensorflow.keras.utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

import numpy as np
predictions=np.round (model.predict (x))
print (predictions)
