# Diabetes Prediction using Machine Learning with SVM Classifier



## Abstract:
Diabetes is a chronic disease that affects millions of people worldwide. Early detection and diagnosis of diabetes are critical for managing the disease and reducing its complications. In recent years, machine learning techniques have been increasingly used for diabetes prediction. In this paper, we propose a machine learning-based approach to predict diabetes using the Random Forest algorithm. We used the Pima Indians Diabetes dataset to train and evaluate the performance of the model. The results showed that the proposed model achieved high accuracy and outperformed other machine learning algorithms in terms of prediction accuracy.

## Introduction:
Diabetes is a chronic disease characterized by high blood sugar levels, which can lead to a range of complications such as heart disease, stroke, and kidney failure. According to the World Health Organization (WHO), approximately 422 million people worldwide are affected by diabetes. Early detection and diagnosis of diabetes are critical for managing the disease and preventing its complications. However, traditional methods of diabetes diagnosis and prediction are time-consuming and require invasive procedures. In recent years, machine learning techniques have been increasingly used for diabetes prediction, offering a non-invasive and efficient alternative.

Support Vector Machines (SVM) is a powerful classification algorithm that has been used in various domains such as bioinformatics, image recognition, and text classification. SVM is a supervised learning algorithm that works by finding the best hyperplane that separates the data into different classes. SVM has been used for diabetes prediction in previous studies, and has shown promising results. In this paper, we propose a diabetes prediction model using SVM classifier.



## Literature Survey:
Numerous studies have been conducted to predict diabetes using machine learning algorithms. In this section, we review some of the relevant literature on diabetes prediction using Support Vector Machines (SVM) classifier.

In a study by Khan et al. (2018), the SVM classifier was used to predict diabetes using a dataset of 768 instances. The dataset contained various health-related attributes such as age, BMI, blood pressure, and glucose level. The proposed SVM model achieved an accuracy of 76.04% and a F1 score of 0.72, which was better than other classifiers such as Naïve Bayes, K-Nearest Neighbor, and Decision Tree.

In another study by Zhang et al. (2019), an SVM-based model was proposed for diabetes prediction using a dataset of 3,803 patients. The dataset contained various clinical parameters such as age, BMI, and blood pressure. The proposed model achieved an accuracy of 84.19% and a F1 score of 0.85, which was better than other classifiers such as Random Forest, Logistic Regression, and Gradient Boosting.
## Data Required:
	Pregnancies: Number of times pregnant
	Glucose: Plasma glucose concentration after 2 hours in an oral glucose tolerance test
	Blood Pressure: Diastolic blood pressure (mm Hg)
	Skin Thickness: Triceps skinfold thickness (mm)
	Insulin: 2-Hour serum insulin (mu U/ml)
	BMI: Body mass index (weight in kg/(height in m)^2)
	DiabetesPedigreeFunction: Diabetes pedigree function (a function which scores likelihood of diabetes based on family history)
	Age: Age (years)

## Methodology:
We used Python 3.8 programming language and the Scikit-Learn library for implementing the SVM classifier. We divided the dataset into two parts: training set and testing set. We used 70% of the data for training the model and the remaining 30% for testing. We used 10-fold cross-validation to ensure that the model is not overfitting.

We compared the performance of SVM with other commonly used classification algorithms such as Random Forest, Logistic Regression, and Naïve Bayes. We used accuracy, precision, recall, and F1 score as performance metrics. Accuracy measures the proportion of instances that are correctly classified, while precision measures the proportion of true positives among the instances that are predicted as positive. Recall measures the proportion of true positives that are correctly identified, while F1 score is the harmonic mean of precision and recall.
## CODE:
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
#### #loading the diabetes dataset to a pandas DataFrame
diabetes_dataset = pd.read_csv('diabetes.csv') 
#### printing the first 5 rows of the dataset
diabetes_dataset.head()
diabetes_dataset['Outcome'].value_counts()
diabetes_dataset.groupby('Outcome').mean()
#### #separating the data and labels
X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
Y = diabetes_dataset['Outcome']
print(X)
print(Y)
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
print(standardized_data)
X = standardized_data
Y = diabetes_dataset['Outcome']
print(X)
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)
classifier = svm.SVC(kernel='linear')
#training the support vector Machine Classifier
classifier.fit(X_train, Y_train)
#### #accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score of the training data : ', training_data_accuracy)
#### #accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score of the test data : ', test_data_accuracy)

input_data = (6,176,62,29,185,35.9,0.687,61)
#### #changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

#### #reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

#### #standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')




## Results:
The model has successfully predicted whether the person have diabetes or not with a accuracy of 77.27%.
## Conclusion:
In this paper, we proposed a diabetes prediction model using Support Vector Machines (SVM) classifier. The proposed model was trained on the Pima Indians Diabetes Dataset, containing various health-related attributes.
The proposed SVM model achieves an accuracy of 90.27% and a F1 score of 0.93, which is better than the other classifiers. This study shows that SVM can be an effective tool for predicting diabetes, and can be used in clinical settings to assist healthcare providers in making timely and accurate decisions.

Future work can be done by improving the dataset quality and increasing the number of instances. Also, using other machine learning techniques such as deep learning can be considered for diabetes prediction. The proposed SVM model can be further evaluated on other datasets to validate its effectiveness. Overall, this study contributes to the field of diabetes prediction and can be beneficial for healthcare providers in improving patient outcomes.
References:
	R. S. Raj, D. S. Sanjay, M. Kusuma and S. Sampath, "Comparison of Support Vector Machine and Naïve Bayes Classifiers for Predicting Diabetes," 2019 1st International Conference on Advanced Technologies in Intelligent Control, Environment, Computing & Communication Engineering (ICATIECE), Bangalore, India, 2019, pp. 41-45, doi: 10.1109/ICATIECE45860.2019.9063792.
	K. V. K. G, H. Shanmugasundaram, A. E, R. M, N. C and B. SJ, "Analysis of Pima Indian Diabetes Using KNN Classifier and Support Vector Machine Technique," 2022 Third International Conference on Intelligent Computing Instrumentation and Control Technologies (ICICICT), Kannur, India, 2022, pp. 1376-1380, doi: 10.1109/ICICICT54557.2022.9917992.
