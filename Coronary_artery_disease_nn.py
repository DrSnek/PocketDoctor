import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

import itertools
from sklearn.impute import SimpleImputer

from sklearn import model_selection
from sklearn.neural_network import MLPClassifier
from sklearn import svm

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler


url = "https://drive.google.com/file/d/1T_kvUBOuoVLku_N6Am1dxb1M2qZUBFf4/view?usp=sharing"
url='https://drive.google.com/uc?id=' + url.split('/')[-2]
df = pd.read_csv(url)
df.head()


df.info()
df.describe().transpose()


fig, ax = plt.subplots(figsize=(20,15)) 
corrMatrix = pd.DataFrame(df, columns=['age', 'cigsPerDay', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose', "TenYearCHD"]).corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()


sns.countplot(x="TenYearCHD", data=df, palette="pastel").set_title("Target Class Distribution")
sns.despine()
df["TenYearCHD"].value_counts()


df.info()
df.dropna(inplace = True)
df.info()


X = df.drop("TenYearCHD", axis = 1)
Y = df["TenYearCHD"].copy()

print("X shape before processing is", X.shape)
df_numerical = X.drop(["male", "education", "currentSmoker", "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes"], axis = 1)

numerical_pipeline = Pipeline([
                               ("std_scaler", StandardScaler())
                               ])

numerical_features = list(df_numerical)
categorical_features = ["male", "education", "currentSmoker", "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes"]

full_pipeline = ColumnTransformer([
                ("numerical", numerical_pipeline, numerical_features), 
                ("categorical", OneHotEncoder(), categorical_features)
                ])
X_prepared = full_pipeline.fit_transform(X)

print("X shape after processing is", X_prepared.shape)
np.unique(np.isnan(X_prepared), return_counts=True)


X_train_im, X_test, y_train_im, y_test = train_test_split(X_prepared, Y, test_size = 0.2, random_state = 0)


undersampler = RandomUnderSampler(sampling_strategy="majority", random_state=0)
X_train, y_train = undersampler.fit_resample(X_train_im, y_train_im)

df_target_sm = pd.DataFrame(y_train, columns=["TenYearCHD"])
sns.countplot(x="TenYearCHD", data=df_target_sm, palette="pastel").set_title("Training Data Afterwards")
sns.despine()


df.isnull().sum().transpose()


df.info()
