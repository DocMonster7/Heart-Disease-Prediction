import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


dataset = pd.read_csv('heart.csv')
# user_data =pd.read_csv('user_data.csv')
st.title("Heart Disease Prediction")

age = st.number_input('AGE',step=1)
sex = st.selectbox(
     'SEX',
     ('Male', 'FEMALE'))
chest_pain=st.selectbox('Chest Pain Type',('Asymptomatic','Atypical Angina','Non-Anginal pain','Typical Angina'))
rest_blood = st.number_input('Resting Blood Pressure',step=1)
cholestrol=st.number_input('Serum Cholestrol (mg/dl)',step=1)
blood_sugar = st.selectbox("Fasting Blood Sugar",('<120mg/dl','>120mg/dl'))
egc = st.selectbox("Resting electrocardiographic results:",('Normal','Having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)','Showing probable or definite left ventricular hypertrophy by Estes criteria'))
max_heart = st.number_input("Max HeartRate achived",step=1)
exercise = st.selectbox("Exercise Induced Agina",("Yes","No"))
depression = st.slider("ST depression induced by exercise relative to rest", min_value=0.00,max_value=5.00,step=0.01)
slope = st.selectbox("The slope of the peak exercise ST segment",("Upsloping","Flat","Downsloping"))
major_vessel = st.selectbox("number of major vessels (0-3) colored by flourosopy",(0,1,2,3))
thalium = st.selectbox("Thalium Stress Test Result",("Fixed Defect",'Normal','Reversible Defect'))
# st.header("Dataset Description")
# st.markdown('This data set dates from 1988 and consists of four databases: Cleveland, Hungary, Switzerland, and Long Beach V. It contains 76 attributes, including the predicted attribute, but all published experiments refer to using a subset of 14 of them. The "target" field refers to the presence of heart disease in the patient. It is integer valued 0 = no disease and 1 = disease.')
# st.dataframe(dataset)

# rcParams['figure.figsize'] = 20, 14
# plt.matshow(dataset.corr())
# plt.yticks(np.arange(dataset.shape[1]), dataset.columns)
# plt.xticks(np.arange(dataset.shape[1]), dataset.columns)
# plt.colorbar()

# st.pyplot(plt,True)

user_data =pd.DataFrame()

# Data Processing
dataset = pd.get_dummies(dataset, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
standardScaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])

y = dataset['target']
# st.dataframe(y)
X = dataset.drop(['target'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)
st.dataframe(X_train    )
user_data = user_data.append(X_test.tail(1))
st.dataframe(user_data)
knn = KNeighborsClassifier(n_neighbors = 8)
knn.fit(X_train,y_train)
# y_pred=knn.predict(user_data)
# if y_pred == 1:
#     st.warning("Congestive Heart Failure Emminent")
# if y_pred == 0:
#     st.success("Rejoice my person")
# user_data= pd.get_dummies(user_data, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
# user_data[columns_to_scale] = standardScaler.fit_transform(user_data[columns_to_scale])

# st.dataframe(dataset)
# st.dataframe(user_data)
# Machine Learning

# st.dataframe(X)
# st.dataframe(y)
# st.code(X_test.shape) 
# st.dataframe(user_data.tail(1))
# X_test.append(user_data)
# st.edataframe(X_test.tail(1))


# # st.dataframe(user_data)
# ## K neigbor Classification
# # st.header('K-Neighbour Classification')
# # knn_scores = []
# # for k in range(1,21):
# #     knn_classifier = KNeighborsClassifier(n_neighbors = k)
# #     knn_classifier.fit(X_train, y_train)
# #     # y_pred=knn_classifier.predict(user_data)
# #     knn_scores.append(knn_classifier.score(X_test, y_test))

# # pred = knn_classifier.predict(user_data.tail(1))
# # st.code(pred)
# # st.markdown("The score for K Neighbors Classifier is {}% with {} nieghbors.".format(round(knn_scores[7]*100,2), 8))

# knn = KNeighborsClassifier(n_neighbors = 8)
# knn.fit(X_train,y_train)
# y_pred=knn.predict(user_data)
# if y_pred == 1:
#     st.warning("Congestive Heart Failure Emminent")
# if y_pred == 0:
#     st.success("Rejoice my person")
# st.code(y_pred)
# st.dataframe(X_test)
# st.dataframe(user_data)
# st.code(y_pred)
# plt.plot([k for k in range(1, 21)], knn_scores, color = 'red')
# for i in range(1,21):
#     plt.text(i, knn_scores[i-1], (i, round(knn_scores[i-1],2)))
# plt.xticks([i for i in range(1, 21)])
# plt.xlabel('Number of Neighbors (K)')
# plt.ylabel('Scores')
# plt.title('K Neighbors Classifier scores for different K values')

# st.pyplot(plt,True)







# #Support Vector Classifier
# st.header('Support Vector Classifier')

# svc_scores = []
# kernels = ['linear', 'poly', 'rbf', 'sigmoid']
# for i in range(len(kernels)):
#     svc_classifier = SVC(kernel = kernels[i])
#     svc_classifier.fit(X_train, y_train)
#     svc_scores.append(svc_classifier.score(X_test, y_test))

# st.markdown("The score for Support Vector Classifier is {}% with {} kernel.".format(round(svc_scores[0]*100,2), 'linear'))


# colors = rainbow(np.linspace(0, 1, len(kernels)))
# plt.bar(kernels, svc_scores, color = colors)
# for i in range(len(kernels)):
#     plt.text(i, svc_scores[i], round(svc_scores[i],2))
# plt.xlabel('Kernels')
# plt.ylabel('Scores')
# plt.title('Support Vector Classifier scores for different kernels')

# st.pyplot(plt,True)

# #Decision Tree Classifier
# st.header("Decision Tree Classifier")

# dt_scores = []
# for i in range(1, len(X.columns) + 1):
#     dt_classifier = DecisionTreeClassifier(max_features = i, random_state = 0)
#     dt_classifier.fit(X_train, y_train)
#     dt_scores.append(dt_classifier.score(X_test, y_test))

# st.markdown("The score for Decision Tree Classifier is {}% with {} maximum features.".format(round(dt_scores[17]*100,2), [2,4,18]))


# plt.plot([i for i in range(1, len(X.columns) + 1)], dt_scores, color = 'green')
# for i in range(1, len(X.columns) + 1):
#     plt.text(i, dt_scores[i-1], (i, round(dt_scores[i-1],2)))
# plt.xticks([i for i in range(1, len(X.columns) + 1)])
# plt.xlabel('Max features')
# plt.ylabel('Scores')
# plt.title('Decision Tree Classifier scores for different number of maximum features')

# st.pyplot(plt,True)

# #Random Forest Classifier

# st.header("Random Forest Classifier")

# rf_scores = []
# estimators = [10, 100, 200, 500, 1000]
# for i in estimators:
#     rf_classifier = RandomForestClassifier(n_estimators = i, random_state = 0)
#     rf_classifier.fit(X_train, y_train)
#     rf_scores.append(rf_classifier.score(X_test, y_test))

# st.markdown("The score for Random Forest Classifier is {}% with {} estimators.".format(round(rf_scores[1]*100,2), [100, 500]))

# colors = rainbow(np.linspace(0, 1, len(estimators)))
# plt.bar([i for i in range(len(estimators))], rf_scores, color = colors, width = 0.8)
# for i in range(len(estimators)):
#     plt.text(i, rf_scores[i], round(rf_scores[i],2))
# plt.xticks(ticks = [i for i in range(len(estimators))], labels = [str(estimator) for estimator in estimators])
# plt.xlabel('Number of estimators')
# plt.ylabel('Scores')
# plt.title('Random Forest Classifier scores for different number of estimators')

# st.pyplot(plt,True)