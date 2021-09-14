import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
from sklearn import datasets
from sklearn.cluster import KMeans
# from sklearn.datasets import load_files
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import PolynomialFeatures


#load data
path = r"processed.cleveland.data"
headernames = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
data = pd.read_csv(path, names=headernames )

print(data)

new_data = data[['age', 'sex', 'num']]


#analysis
a = data.shape #dimensions
b = data.groupby('num').size()
c = data.dtypes
correlations = data.corr(method='pearson')
sns.heatmap(correlations, annot=True)
plt.show()

scatter_diagram = sns.pairplot(data, hue='num')
plt.show()
density = data.plot(kind='density', subplots=True, sharex=False)
plt.show()

####################### classification

X = data.loc[:, 'oldpeak'].values.reshape(-1, 1) #independent columns
y = data.loc[:, 'num'].values.reshape(-1, 1) #target column 

X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.20,random_state=1)


clf = DecisionTreeClassifier()
clf = clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)

con_f = confusion_matrix(Y_test, Y_pred)

accuracy = accuracy_score(Y_test,Y_pred)

precision = precision_score(y_true=Y_test, y_pred=Y_pred, average=None)

recall = recall_score(y_true=Y_test, y_pred=Y_pred, average=None)

print(accuracy)
print(precision)
print(recall)

####################### clustering
 

km = KMeans(n_clusters=2)
ndata = data.drop(['num'], axis=1)
y_predicted = km.fit_predict(ndata)
# print(y_predicted)
ndata['cluster']=y_predicted
plt.scatter(ndata.values[:, 0], ndata.values[:, 1], c = y_predicted, s = 20, cmap = 'summer')

plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.xlabel('age')
plt.ylabel('sex')
plt.legend()
plt.show()


##################### regression

#create linear model and train it
LR = LinearRegression()
LR.fit(X_train, Y_train)

#use model to predict 
Y_pred = LR.predict(X_test)



fig, ax = plt.subplots(figsize =(14, 9))
ax.scatter(X_test, Y_test, label='actual data')
ax.plot(X_test, Y_pred, alpha=0.7, antialiased=True)
ax.set_ylabel('num', fontsize=14)
ax.set_xlabel(' oldpeak', fontsize=14)

plt.figure(2)
plt.plot(Y_test, label = 'actual')
plt.plot(Y_pred, label='predict')
plt.show()
 
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
pol_reg = LinearRegression()
pol_reg.fit(X_poly, y) 
plt.figure(3)
plt.show()
