# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Data Preparation: Load and handle the dataset, separating features and target variables.
2. Data Splitting: Split the dataset into training and testing sets.
3. Feature Engineering: Transform text data into numerical feature vectors.
4. Model Building and Evaluation: Initialize SVM classifier, train the model, and predict target labels for evaluation.


## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Gokul Nath R
RegisterNumber:  212224230077
*/

import pandas as pd
data=pd.read_csv('C:/Users/admin/Downloads/printed pdfs/spam.csv',encoding="Windows-1252")
from sklearn.model_selection import train_test_split

data

data.shape

x=data['v2'].values

y=data['v1'].values

x.shape

y.shape

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.35, random_state=0)

x_train

x_train.shape

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)

from sklearn.svm import SVC

svc=SVC()

svc.fit(x_train, y_train)

y_pred=svc.predict(x_test)

y_pred

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

acc=accuracy_score(y_test, y_pred)
acc

con=confusion_matrix(y_test,y_pred)
print(con)

cl=classification_report(y_test,y_pred)
print(cl)
```

## Output:


<img width="744" height="448" alt="329268661-0a86ceb6-1c87-4108-a09b-7172f60caed3" src="https://github.com/user-attachments/assets/d5579942-7220-4e68-aa22-48a063a845ca" />


## X-Train


<img width="1239" height="215" alt="329269326-9cce3e36-5496-4282-9255-882010f62405" src="https://github.com/user-attachments/assets/3e912b74-14c4-489b-ae13-521fedd7987a" />

## Y-Pred

<img width="691" height="39" alt="329269685-000a6f61-9369-4c80-8755-fb9c2cf83459" src="https://github.com/user-attachments/assets/fed3014e-45ce-4529-a812-4b9a9ab281e9" />

## Accuracy


<img width="201" height="38" alt="329269856-923d92dd-5af8-4c68-b2bb-f3c1a3d41b9b" src="https://github.com/user-attachments/assets/74d2631a-0fed-4ac5-be43-6f3adbacc9b7" />

## Confusion matrix


<img width="153" height="67" alt="329269965-265e3ddd-3790-425e-972b-b52059b10e99" src="https://github.com/user-attachments/assets/4efff438-011f-42a3-9761-2f21322ae7b4" />


## Classification Matrix


<img width="601" height="187" alt="329270110-5d5e0cfb-97be-4ad3-9eb9-13c756f82ea7" src="https://github.com/user-attachments/assets/9436d7d7-53ba-40c6-bcad-796dee67c973" />


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
