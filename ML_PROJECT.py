import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as knc
from sklearn.model_selection import train_test_split

#Data Preprossesing

#reading the data
training_data = pd.read_csv('train.csv')

print(training_data.shape)
#exploring the data 
#print(training_data.info())
#print(training_data.isna().sum()) 

#--> the number of missing values in age is not that big so we can fill these values with the mean 
#--> the number of missing values in cabin is more than 50% -->col. will be droped
#--> the two missing values in Embarked column will be droped
#--> ticket & name is uniqu so there is no point of having them --> col.s will be droped

#print(training_data['Age'].mean())

#calculating the mean 
mean_of_age = training_data['Age'].mean()

#filling the Age column with the mean
training_data['Age'] = training_data['Age'].fillna(mean_of_age)

#droping the Cabin, name and ticket col
training_data = training_data.drop(['Cabin','Name','Ticket'],axis=1)

#droping the missng rows
training_data = training_data.dropna()

#print(training_data.shape)
#print(training_data.head(10))

#is there any outlier in this data ? --> only one way to find out
np.random.seed(0)

f = np.random.randn(len(training_data['Fare']))
plt.scatter(training_data['Fare'],f)

z = np.random.randn(len(training_data['Age']))
plt.scatter(training_data['Age'],z)

#plt.show()


# age is clean but there is 3 outliers in (Fare)
#print(training_data[training_data['Fare']>400])


#it is weird that 3 passengers have the same fare and the same ticket number(512) --> we are gonna drop these rows
training_data = training_data[training_data['Fare']<400]

#print(training_data[training_data['Fare']>400])
print(training_data.shape)
training_data['Sex'] = training_data['Sex'].map({'male': 1, 'female': 0})
print(training_data.head())

#------------------------------------------------------
#KNN HERE WE GOO
x = training_data[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']].values
y = training_data['Survived'].values


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=21, stratify=y)
knn = knc(n_neighbors = 6)
knn.fit(x_train,y_train)
print(knn.score(x_test,y_test))

