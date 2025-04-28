# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.
2.Upload and read the dataset.
3.Check for any null values using the isnull() function.
5.4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: MITHUN G
RegisterNumber:  212223080030
*/
import pandas as pd
data=pd.read_csv("Employee.csv")
data.head()

data.info()

data.isnull().sum()

data['left'].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['salary']=le.fit_transform(data['salary'])
data.head()

x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
x.head()

y=data['left']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_predict)
accuracy

dt.predict([[0.5,0.8,9,206,6,0,1,2]]
```

## Output:

Data Head:

![image](https://github.com/user-attachments/assets/ac48bd81-2d30-4ffa-bba8-54cea40725ad)


Dataset Info:

![image](https://github.com/user-attachments/assets/5d3fa857-e787-4e35-baaa-fd5abbb1779d)

Null Dataset:

![image](https://github.com/user-attachments/assets/81c0f035-0233-4c65-9728-f6429c1c72dd)

Values Count in Left Column:

![image](https://github.com/user-attachments/assets/e8508ebb-ef87-4e88-91fd-c55fb96c45bf)

Dataset transformed head:

![image](https://github.com/user-attachments/assets/5284ebc3-8a5f-4ed8-938e-733a87564fd1)

X.head():

![image](https://github.com/user-attachments/assets/c8a9510d-14a5-4472-ab04-a7a9d088b8da)

Accuracy:

![image](https://github.com/user-attachments/assets/fa0f75ac-1bcf-422d-89ef-ac283dc77067)

Data Prediction:

![image](https://github.com/user-attachments/assets/30f0a14c-22e3-4a50-bd43-02d3d322826f)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
