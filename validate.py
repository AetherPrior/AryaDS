import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


df = pd.read_csv('training_set.csv',index_col=0)
X = np.array(df.drop(columns='Y'))
y = df['Y']

with open('model.pkl','rb') as f:
    model_dict = pickle.load(f)

pca = model_dict['pca']
svc = model_dict['classifier']

X = (X - np.mean(X,axis=0))/np.std(X,axis=0)
X = pca.transform(X)

acc_ = 0
n_splits=5
for i in range(n_splits):
    x_train, x_val, y_train, y_val = train_test_split(X,y,train_size=0.8)
    y_pred = svc.predict(x_val)
    acc_ += accuracy_score(y_val, y_pred)

print(acc_/n_splits)

