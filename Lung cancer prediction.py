
import pandas as pd
import matplotlib as mpl
import seaborn as sns 
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from keras import backend as K
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1
import matplotlib.pyplot as plt2
import plotly.express as px
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression,RidgeClassifier,SGDClassifier,PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC,LinearSVC,NuSVC
from sklearn.neighbors import KNeighborsClassifier,NearestCentroid
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB,BernoulliNB
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_score,accuracy_score
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV,RepeatedStratifiedKFold
from sklearn.metrics import recall_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import metrics



data = pd.read_csv("G:\Graduation project/Dataset.csv")

data.head()



data.describe()


plt.style.use("seaborn")
data.hist(figsize=(25,25), bins=15)




plt.figure(figsize=(9,9))
sns.histplot(data[['AGE', 'LUNG_CANCER']], x = "AGE", hue="LUNG_CANCER")
plt.title("")
plt.show()



plt.figure(figsize=(9,9))
sns.histplot(data[['AGE', 'GENDER']], x = "AGE", hue="GENDER")
plt.title("")
plt.show()


plt.style.use("seaborn")
plt.figure(figsize=(9,9))
sns.histplot(data[['GENDER', 'LUNG_CANCER']], x = "GENDER", hue="LUNG_CANCER")
plt.title("")
plt.show()



plt.style.use("seaborn")
plt.figure(figsize=(15,8))
plt.title("Genders", fontsize=20, y=1.02)
sns.countplot(x = data.GENDER ,palette="crest")
plt.show()




data["GENDER"] = data["GENDER"].map({"F": 0, "M": 1})
data['LUNG_CANCER']= data['LUNG_CANCER'].map({'NO':0, "YES":1})



data.info()

CollectedData = []

# X data 
X = data.drop("LUNG_CANCER", axis = 1)




# y data 
y = data["LUNG_CANCER"]
print(y)
y.head()




x_train, x_test, y_train, y_test = train_test_split (X, y , test_size = 0.2, random_state = 10, stratify=y)


len(x_test), len(x_train)


plt.figure(figsize=(20,8))
plt.title("Data before Scaling", fontsize = 20, y=1.0)
sns.boxenplot(data=x_train)
plt.show()


# Scaling data:
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
print(x_test.shape)





plt.figure(figsize=(20,8))
plt.title("Data after Scaling", fontsize = 20, y=1.0)
sns.boxenplot(data=x_train)
plt.show()


from warnings import filterwarnings
filterwarnings('ignore')



print('LR')
LR = LogisticRegression();
LR.fit(x_train, y_train)
model1=LR.predict(x_test)
print(LR.score(x_train,y_train))
print(classification_report(y_test,model1))
print('ffffffffffffff')

print('SVM')
SVM = SVC()
SVM.fit(x_train, y_train)
model2=SVM.predict(x_test)
print(SVM.score(x_train,y_train))
print(classification_report(y_test,model2))

print('KNN')
knn = KNeighborsClassifier(n_neighbors=67)
knn.fit(x_train, y_train)
model3=knn.predict(x_test)
print(knn.score(x_train,y_train))
print(classification_report(y_test,model3))

o=10
while o>0:
 YourData = []
 YourGender = input('Enter your gender: ')
 if YourGender=='m':
     YourGender = 1
 else:
     YourGender = 0
 YourData.append(YourGender)
 
 YourAge = input('Enter your age: ')
 YourData.append(YourAge)
 
 DoYouSmoke = input('Do you smoke? (y/n) ')
 if DoYouSmoke=='y':
     DoYouSmoke = 1
 else:
     DoYouSmoke = 0
 YourData.append(DoYouSmoke)
 
 YellowFingers = input('Do you have yellow fingers?(y,n) ')
 if YellowFingers=='y':
     YellowFingers = 1
 else:
     YellowFingers = 0
 YourData.append(YellowFingers)
 
 Anxiety = input('Do you feel anxiety?(y,n) ')
 if Anxiety=='y':
     Anxiety = 1
 else:
     Anxiety = 0
 YourData.append(Anxiety)
 
 PeerPressure = input('Do you have peer pressure? (y,n) ')
 if PeerPressure == 'y':
     PeerPressure = 1
 else:
     PeerPressure = 0
 YourData.append(PeerPressure)
 
 ChronicDisease = input('Do you have chronic diseasse? (y,n) ')
 if ChronicDisease == 'y':
     ChronicDisease = 1
 else:
     ChronicDisease = 0
 YourData.append(ChronicDisease)
 
 Fatigue = input('Do u feel any fatigue? (y,n) ') 
 if Fatigue == 'y':
     Fatigue = 1
 else:
     Fatigue = 0
 YourData.append(Fatigue)
 
 Allergy = input('Do u have any allergy? (y,n) ')
 if Allergy == 'y':
     Allergy = 1
 else:
     Allergy = 0
 YourData.append(Allergy)
 
 Wheezing = input('Do u have wheezing? (y,n) ')
 if Wheezing == 'y':
     Wheezing = 1
 else:
     Wheezing = 0
 YourData.append(Wheezing)
 
 AlcoholConsuming = input('Do u have AlcoholConsuming? (y,n) ')
 if AlcoholConsuming == 'y':
     AlcoholConsuming = 1
 else:
     AlcoholConsuming = 0
 YourData.append(AlcoholConsuming)
 
 Coughing = input('Do you cough? (y,n) ')
 if Coughing == 'y':
     Coughing = 1
 else:
     Coughing = 0
 YourData.append(Coughing)
 
 Shortnessofbreath = input('Do you face a shortness of breath? (y,n) ')
 if Shortnessofbreath == 'y':
     Shortnessofbreath = 1
 else:
     Shortnessofbreath = 0
 YourData.append(Shortnessofbreath)
 
 SwallowingDifficulty=input('Do you face a swallowing difficulty? (y,n) ')
 if SwallowingDifficulty == 'y':
     SwallowingDifficulty = 1
 else:
     SwallowingDifficulty = 0
 YourData.append(SwallowingDifficulty)
 
 ChestPain=input('Do you face a chest pain? (y,n) ')
 if ChestPain == 'y':
     ChestPain = 1
 else:
     ChestPain = 0
 YourData.append(ChestPain)
 YourData = np.reshape(YourData, (1,15))
 print(YourData)
 
 
 x = data.iloc[:, [0,1,2, 3,4,5,6,7,8,9,10,11,12,13,14]].values
 y = data["LUNG_CANCER"].values
 x_train, x_test, y_train, y_test = train_test_split(x, y ,test_size=0.2 , random_state=10)
 model = LogisticRegression()
 model.fit(x_train, y_train)
 model.predict(x_train)
 
 model_predict = model.predict(x_test)
 
 model_pred =model.predict(YourData )
 print(model.score(x_test,y_test))
 if model_pred == 1:
     result = "Have characteristics of lung cancer"
 elif model_pred == 0 :
     result = "Doesn't have characteristics of lung cancer"
 else :
     result = "eror"
 
 
 print("Result with accuracy ",model.score(x_test,y_test)*100 ,'%' ,result)






