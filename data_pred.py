import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


df = pd.read_csv('test_set.csv',index_col=0)
X = np.array(df)

with open('model.pkl','rb') as f:
    model_dict = pickle.load(f)

pca = model_dict['pca']
svc = model_dict['classifier']

X = (X - np.mean(X,axis=0))/np.std(X,axis=0)
X = pca.transform(X)

y_pred = svc.predict(X)
df['Y'] = y_pred

df.to_csv('predictions.csv',index=False)