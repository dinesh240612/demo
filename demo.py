# In[ ]:
#importing required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
# In[ ]:
train_data = pd.read_csv("data/cs-training.csv")
train_data.head()
# In[ ]:
#Let us remove unnamed column because it is representing only index values and does not impact the model.
train_data = pd.read_csv("data/cs-training.csv").drop(['Unnamed:0'],axis=1)
train_data.head()
# In[ ]:
train_data.duplicated()
# In[ ]:
train_data.duplicated().sum()
# In[ ]:
train_data.duplicated().value_counts()
# In[ ]:
trainset_no_duplicates = train_data.drop_duplicates()
# In[ ]:
trainset_no_duplicates.duplicated().sum()
# In[ ]:
trainset_no_duplicates.isnull()

# In[ ]:
trainset_no_duplicates.isnull().sum()
# In[ ]:
round(trainset_no_duplicates.isnull().sum()/trainset_no_duplicates.shape[0]*100,1)
# In[ ]:
#Function to create a missing values
def findmissvalues(df):
    return round(df.isnull().sum()/df.shape[0]*100,1)

# In[ ]:
findmissvalues(trainset_no_duplicates)
# In[ ]:
#Firstly we are handling NumberOfDependents column.
trainset_no_duplicates.NumberOfDependents.isnull().sum()
# In[ ]:
trainset_no_duplicates[trainset_no_duplicates.NumberOfDependents.isnull()]

# In[ ]:
#Let us check for Number of Dependents and Monthly Income relationship
trainset_no_duplicates[trainset_no_duplicates.NumberOfDependents.isnull()].describe()
# In[ ]:
trainset_no_duplicates[trainset_no_duplicates.MonthlyIncome.isnull()].describe()
# In[ ]:
#Let us treat Number of Dependents with aggregate function mode
trainset_no_duplicates['NumberOfDependents'].agg(['mode'])
# In[ ]:
trainset_no_duplicates.groupby(['NumberOfDependents']).size()
# In[ ]:
#Let us divide the data frame in two, one with null values in NumberOfDependents and Monthly income. Other data frame has rest of the all
data_depmiss=trainset_no_duplicates[trainset_no_duplicates.NumberOfDependents.isnull()]
data_nodepmiss=trainset_no_duplicates[trainset_no_duplicates.NumberOfDependents.notnull()]
# In[ ]:
data_depmiss.shape
# In[ ]:
data_nodepmiss.shape
# In[ ]:
#Filling all the missing values in data_depmiss with 0
data_depmiss['NumberOfDependents'] = data_depmiss['NumberOfDependents'].fillna(0)
data_depmiss['MonthlyIncome'] = data_depmiss['MonthlyIncome'].fillna(0)
# In[ ]:
#Let us check where there are any missing values in data_depmiss.
data_depmiss.NumberOfDependents.isnull().sum()
# In[ ]:
data_depmiss.MonthlyIncome.isnull().sum()
# In[ ]:
findmissvalues(data_depmiss)
# In[ ]:
#Let us start working on rest of the people who not mention about their monthly income.
findmissvalues(data_nodepmiss)
# In[ ]:
data_nodepmiss['MonthlyIncome'].agg(['mean','median','min'])
# In[ ]:
data_nodepmiss['MonthlyIncome'].agg(['max'])
# In[ ]:
#let us fill null values with median.
data_nodepmiss['MonthlyIncome'] = data_nodepmiss['MonthlyIncome'].fillna(data_nodepmiss['MonthlyIncome'].median())
# In[ ]:
findmissvalues(data_nodepmiss)
# In[ ]:
#Let us merge both the data frames.
training_dataset= data_nodepmiss.append(data_depmiss)
# In[ ]:
training_dataset.shape
# In[ ]:
#Let us finally check once again for null values.
findmissvalues(training_dataset)
# In[ ]:
training_dataset.head()
# In[ ]:
#finding ratio of defaulters and non defaulters
training_dataset.groupby(['SeriousDlqin2yrs']).size()/ training_dataset.shape[0]*100
# In[ ]:
#Let us start working on each and every feature.
training_dataset.RevolvingUtilizationOfUnsecuredLines.describe()
# In[ ]:
(training_dataset[training_dataset['RevolvingUtilizationOfUnsecuredLines']>10]).groupby(['SeriousDlqin2yrs']).size()
# In[ ]:
(training_dataset[training_dataset['RevolvingUtilizationOfUnsecuredLines']>10]).describe()
# In[ ]:
# Let us remove the rows in which RevolvingUtilizationOfUnsecuredLines is greater than 10.
training_dataset_rm= training_dataset.drop((training_dataset[training_dataset['RevolvingUtilizationOfUnsecuredLines']>10]).index)
# In[ ]:
training_dataset_rm.shape
# In[ ]:
training_dataset_rm.head()
# In[ ]:
# Let us Plot a box_plot to describe about age.
sns.boxplot(training_dataset_rm['age'])
# In[ ]:
training_dataset_rm.groupby(['NumberOfTime30-59DaysPastDueNotWorse']).size()
# In[ ]:
training_dataset_rm.groupby(['NumberOfTime60-89DaysPastDueNotWorse']).size()
# In[ ]:
training_dataset_rm.groupby(['NumberOfTimes90DaysLate']).size()
# In[ ]:
# Let us Treat the outliers in above features.
training_dataset_rm[training_dataset_rm['NumberOfTimes90DaysLate']>=96].groupby(['SeriousDlqin2yrs']).size()
# In[ ]:
training_dataset_rm[training_dataset_rm['NumberOfTimes90DaysLate']>=96]
# In[ ]:
training_dataset_rm['DebtRatio'].describe()
# In[ ]:
# Let us Plot kdeplot for DebtRatio.
sns.kdeplot(training_dataset_rm['DebtRatio'])
# In[ ]:
# Let us find the quantile where debt ratio is equal to median.
training_dataset_rm['DebtRatio'].quantile([.978])
# In[ ]:
# Let us find the how many defaulters are there above 0.978 quantile.
training_dataset_rm[training_dataset_rm['DebtRatio']>3696].groupby(['SeriousDlqin2yrs']).size()
# In[ ]:
training_dataset_rm[training_dataset_rm['DebtRatio']>3696][['SeriousDlqin2yrs','MonthlyIncome']].describe()
# In[ ]:
# Let us check the people who are having DebtRatio > 3696 and target column and Monthly income is equal.
training_dataset_rm[(training_dataset_rm['DebtRatio']>3696) & (training_dataset_rm['SeriousDlqin2yrs'] == training_dataset_rm['MonthlyIncome'])].describe()
# In[ ]:
# Let us create a temporary data frame.
temp_dataset=training_dataset_rm[(training_dataset_rm['DebtRatio']>3696) & (training_dataset_rm['SeriousDlqin2yrs'] == training_dataset_rm['MonthlyIncome'])]
# In[ ]:
# Let us check how many defaulters are there in temporary dataset.
temp_dataset.groupby(['SeriousDlqin2yrs']).size()
# In[ ]:
# Let us drop the temporary data frame since it has only less defaulters and save as new data frame.
train_dataset=training_dataset_rm.drop(training_dataset_rm[(training_dataset_rm['DebtRatio']>3492) & (training_dataset_rm['SeriousDlqin2yrs'] == training_dataset_rm['MonthlyIncome'])].index)
# In[ ]:
train_dataset.shape
# In[ ]:
train_dataset.groupby(['SeriousDlqin2yrs']).size()
# In[ ]:
# Let us Install xgboost
#pip install xgboost
# In[ ]:
# Let us import the libraries
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
# In[ ]:
# Let us Split the data frame.
x= train_dataset.drop(['SeriousDlqin2yrs'],axis=1)
y= train_dataset['SeriousDlqin2yrs']
y.head()
# In[ ]:
X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.2,random_state=42)
# In[ ]:
import xgboost as xgb
from xgboost import XGBClassifier
# In[ ]:
# Model Training.
xg_model = XGBClassifier()
xg_model.fit(X_train,Y_train)
# In[ ]:
# Model Prediction.
xgb_pred = xg_model.predict(X_test)
# In[ ]:
# Calculating Metrics
xgb_accuracy = accuracy_score(Y_test, xgb_pred)
print('xgboost model accuracy:',xgb_accuracy)
# In[ ]:
# Calculate precision
precision = precision_score(Y_test, xgb_pred)
print(f'Precision: {precision:.2f}')
# In[ ]:
# Calculate recall
recall = recall_score(Y_test, xgb_pred)
print(f'Recall: {recall:.2f}')
# In[ ]:
# Calculate F1-score
f1 = f1_score(Y_test, xgb_pred)
print(f'F1-score: {f1:.2f}')
# In[ ]:
# Generate confusion matrix
conf_matrix = confusion_matrix(Y_test, xgb_pred)
print('Confusion Matrix:')
print(conf_matrix)
# In[ ]:

