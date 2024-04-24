# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import pandas and read the csv file.
2.Import Decision tree classifier.
3.Fit the data in the model
4.Find the accuracy score. 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: NIROSHA S
RegisterNumber:212222230097  
*/
```
```
import pandas as pd
df=pd.read_csv("CSVs/Employee.csv")
df.head()
df.info()
df.isnull().sum()
df['left'].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["salary"]=le.fit_transform(df['salary'])
df.head()
x=df[['satisfaction_level','last_evaluation','number_project','average_montly_hours',
      'time_spend_company','Work_accident','promotion_last_5years','salary']]
x.head()
y=df['left']
from sklearn.model_selection import train_test_split as tts
Xtrain,Xtest,Ytrain,Ytest=tts(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(Xtrain,Ytrain)
Ypred=dt.predict(Xtest)
from sklearn import metrics
accuracy=metrics.accuracy_score(Ytest,Ypred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
## Output:
![image](https://github.com/Niroshassithanathan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121418437/9990e3bc-422d-41db-9f89-c9174ebbd30b)

![image](https://github.com/Niroshassithanathan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121418437/b775ba21-d6e1-4be9-a8fb-56abeb08c978)

![image](https://github.com/Niroshassithanathan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121418437/75692201-d568-4ea8-a040-1957048a04be)

![image](https://github.com/Niroshassithanathan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121418437/5e2d12fb-a930-4f5e-bdce-cdf8cbea1895)

![image](https://github.com/Niroshassithanathan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121418437/5e7caeaa-0acc-49b2-b28e-8daf2e1b21fc)

![image](https://github.com/Niroshassithanathan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121418437/046b98b1-d785-45a0-8eda-a543d67c5517)

![image](https://github.com/Niroshassithanathan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121418437/3961120a-2de5-484a-a817-a67f2bb9d15d)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
