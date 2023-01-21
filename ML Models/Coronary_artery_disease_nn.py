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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

!pip install tflearn

from keras.models import Sequential, Model
from keras.layers import Conv2D, Dropout, MaxPooling2D, Input
from keras.layers import BatchNormalization, Activation, Flatten, Dense
from tensorflow.keras import initializers


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


# Helper Functions

def print_accuracy(y_test, y_pred):
  print("%-12s %f" % ('Accuracy:', metrics.accuracy_score(y_test, y_pred)))
  print("%-12s %f" % ('Precision:', metrics.precision_score(y_test, y_pred,labels=None, pos_label=1, average='binary', sample_weight=None)))
  print("%-12s %f" % ('Recall:', metrics.recall_score(y_test, y_pred,labels=None, pos_label=1, average='binary', sample_weight=None)))
  print()

def draw_confusion_matrix(y_test, y_pred, classes):
  plt.cla()
  plt.clf()
  matrix = confusion_matrix(y_test, y_pred)
  plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
  plt.title("Confusion Matrix")
  plt.colorbar()
  num_classes = len(classes)
  plt.xticks(np.arange(num_classes), classes, rotation=90)
  plt.yticks(np.arange(num_classes), classes)
  fmt = 'd'
  thresh = matrix.max() / 2.
  for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
    plt.text(j, i, format(matrix[i, j], fmt), horizontalalignment="center", color="white" if matrix[i, j] > thresh else "black")
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.tight_layout()
  plt.show()
  print()

def draw_roc_curve(y_test, y_score, title, c="blue", line_width=1):
  fpr_log_reg, tpr_log_reg, thresholds = metrics.roc_curve(y_test, y_score)
  plt.figure(2)
  aucroc = metrics.auc(fpr_log_reg, tpr_log_reg)
  plt.plot(fpr_log_reg, tpr_log_reg, color=c, lw=line_width, label = 'AUC = %0.3f' % aucroc)
  plt.title(title)
  plt.xlabel('False Positive Rates')
  plt.ylabel('True Positive Rates')
  plt.legend(loc = 'lower right')
  plt.show()
  print()

def draw_roc_curve_individual(y_test, y_score, label, line_width=1):
  fpr_log_reg, tpr_log_reg, thresholds = metrics.roc_curve(y_test, y_score)
  aucroc = metrics.auc(fpr_log_reg, tpr_log_reg)
  plt.plot(fpr_log_reg, tpr_log_reg, lw=line_width, label = label + ', AUC = %0.3f' % aucroc)
  plt.xlabel('False Positive Rates')
  plt.ylabel('True Positive Rates')


model = tf.keras.models.Sequential()
model.add(Dense(64, activation='relu', input_dim=24))
model.add(Dropout(0.35))
model.add(Dense(32, activation='relu', input_dim=24))
model.add(Dropout(0.35))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=250, batch_size=10)
y_temp = model.predict(X_test)
nn_y_score = y_temp
y_pred = []
for x in y_temp:
  if x[0] < 0.5:
    y_pred.append(0)
  else:
    y_pred.append(1)
print_accuracy(y_test, y_pred)
draw_confusion_matrix(y_test, y_pred, ["No CAD", "CAD"])
draw_roc_curve(y_test, nn_y_score, "Neural Network - 3 hidden layers ", c="blue", line_width=1)


user_df = pd.DataFrame(user_input, columns=["male", "age", "education", "currentSmoker", "cigsPerDay", "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"])

df = user_backup.append(user_df, ignore_index=True)

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
user_inp = full_pipeline.fit_transform(X)

print("X shape after processing is", user_inp.shape)


print(X_test.shape)
print(user_inp.shape)
print(user_inp)
print(type(X_test))


u_prepared = np.array([user_inp[-1]])

print(u_prepared)
print(u_prepared.shape)

y_temp = model.predict(u_prepared)
print(float(y_temp[0])*100)


