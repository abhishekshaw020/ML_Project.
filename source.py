import pandas as pd
import seaborn as sns
import pickle
df=sns.load_dataset('mpg')
df.isnull().sum()
df.dropna(inplace=True)
X=df[['displacement','horsepower','weight','acceleration']]
Y=df.mpg
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from ctypes import LibraryLoader
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.15,random_state=42)
model=LinearRegression()
model.fit(X_train,Y_train)
filename='mpg_regression.sav'
pickle.dump(model,open(filename,'wb'))