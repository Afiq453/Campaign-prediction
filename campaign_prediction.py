# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 09:36:04 2022

@author: AMD
"""
import os
import pickle
import pandas as pd
import seaborn as sns
import numpy as np
from modules import ModelCreation
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Dense,Dropout,BatchNormalization
from tensorflow.keras import Sequential, Input
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
import scipy.stats as ss
import missingno as msno
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from datetime import datetime
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import EarlyStopping
#%% STATIC
DATA_PATH = os.path.join(os.getcwd(),'dataset','Train.csv')

JOBTYPE_ENCODER_PATH = os.path.join(os.getcwd(),'model','JOBTYPE_ENCODER.pkl')
MARITAL_ENCODER_PATH = os.path.join(os.getcwd(),'model','MARITAL_ENCODER.pkl')
EDUCATION_ENCODER_PATH = os.path.join(os.getcwd(),'model','EDUCATION_ENCODER.pkl')
DEFAULT_ENCODER_PATH = os.path.join(os.getcwd(),'model','DEFAULT_ENCODER.pkl')
HOUSING_ENCODER_PATH = os.path.join(os.getcwd(),'model','HOUSING_ENCODER.pkl')
PERSONAL_ENCODER_PATH = os.path.join(os.getcwd(),'model','PERSONAL_ENCODER.pkl')
COMMUNICATION_ENCODER_PATH = os.path.join(os.getcwd(),'model','COMMUNICATION_ENCODER.pkl')
CAMPAIGN_OUTCOME_ENCODER_PATH = os.path.join(os.getcwd(),'model','CAMPAIGN_OUTCOME_ENCODER.pkl')

SCALER_PATH = os.path.join(os.getcwd(), 'model', 'scaler.pkl')
OHE_PATH = os.path.join(os.getcwd(), 'model', 'ohe.pkl')
KNN_PATH = os.path.join(os.getcwd(), 'model', 'knn.pkl')
MODEL_PATH = os.path.join(os.getcwd(), 'model', 'dl_model.h5')
#%% FUNCTION

def cramers_corrected_stat(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

#%% STEP 1 DATA LOADING
#df = pd.read_csv(DATA_PATH,na_values = 'unknown') # to replace unknows into Nans value
df = pd.read_csv(DATA_PATH)

#%% STEP 2 DATA INSPECTION

df.info()
info = df.describe().T

df.duplicated().sum() # CHECK DUPLICATED # NO DUCPLICATE DATA
df[df.duplicated()]

plt.figure(figsize=(20,20))
df.boxplot()
plt.show()

df.isna().sum() #check Nan Value
msno.matrix(df)# to visualize the NaNs in the data
msno.bar(df)# to visualize the NaNs in the data


df=df.drop(labels='id', axis =1)  #(no function in this data set)
df=df.drop(labels='days_since_prev_campaign_contact', axis =1) # (drop entire columns because to many Nans)
df=df.drop(labels='month', axis =1) 



con_columns = df.columns[(df.dtypes == 'int64') | (df.dtypes == 'float64')]
cat_columns = df.columns[df.dtypes =='object']

for con in con_columns:
    plt.figure()
    sns.distplot(df[con])
    plt.show()


for cat in cat_columns:
    plt.figure()
    sns.countplot(df[cat])
    plt.show()


#%% STEP 3 CLEAN THE DATA


le = LabelEncoder()
paths = [JOBTYPE_ENCODER_PATH,MARITAL_ENCODER_PATH,EDUCATION_ENCODER_PATH,DEFAULT_ENCODER_PATH, 
         HOUSING_ENCODER_PATH,PERSONAL_ENCODER_PATH,
         COMMUNICATION_ENCODER_PATH,CAMPAIGN_OUTCOME_ENCODER_PATH]


for index,i in enumerate(cat_columns):
    temp = df[i]
    temp[temp.notnull()] = le.fit_transform(temp[temp.notnull()])
    df[i] = pd.to_numeric(temp,errors='coerce')
    with open(paths[index],'wb') as file:
        pickle.dump(le,file)

# TO DEALS WITH NaNs
# USING KNNImputer TO DEALS WITH Nans


df_copied = df.copy()
knn_imp = KNNImputer()
df_copied = knn_imp.fit_transform(df_copied)
df_copied = pd.DataFrame(df_copied)

with open(KNN_PATH, 'wb') as file:
    pickle.dump(knn_imp,file)

df_copied.columns = df.columns

#for i in cat_columns:
 #   df_copied[i] = np.floor(df_copied[i]) # #to make sure there is no decimal in categorical data


df_copied.info()
df_copied.isna().sum()

#%% FUTURE SELECTION



#%% Pre Processing

X = df_copied.drop(labels=['prev_campaign_outcome'],axis=1) #feautes
y = df_copied['prev_campaign_outcome']#target

nb_class = len(np.unique(y))


sts = StandardScaler()
X = sts.fit_transform(X)

with open(SCALER_PATH, 'wb') as file:
    pickle.dump(sts,file)


ohe = OneHotEncoder(sparse=False)
y = ohe.fit_transform(np.expand_dims(y,axis=1))

with open(OHE_PATH, 'wb') as file:
    pickle.dump(ohe,file)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,
                                                    random_state=123)


#%% model development (functional / sequential api)

#model = Sequential ()
#model.add(Input((np.shape(X_train)[1],))) 
#model.add(Dense(32, activation = 'relu', name ='Hidden_Layer_1'))
#model.add(BatchNormalization())
#model.add(Dropout(0.2))
#model.add(Dense(32, activation = 'relu', name ='Hidden_Layer_2'))
#model.add(BatchNormalization())
#model.add(Dropout(0.2))
#model.add(Dense(4,activation='softmax', name='Output_layer'))
#model.summary() # to visualize

Model = ModelCreation()
model = Model.simple_lstm_layer(X_train,num_node=128,
                      drop_rate=0.3,output_node=1)

model.compile(loss = 'categorical_crossentropy', optimizer='adam',
                   metrics=['acc'])


LOG_PATH = os.path.join(os.getcwd(),'logs')
log_dir = os.path.join(LOG_PATH, datetime.now().strftime('%Y%m%d-%H%M%S'))
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)


early_stopping_callback = EarlyStopping(monitor='loss', patience = 3)
hist =model.fit(X_train,y_train,
                     validation_data=(X_test,y_test),
                     epochs=10,
                     batch_size=64,
                     callbacks=[tensorboard_callback,early_stopping_callback])


model.save(MODEL_PATH)

#%%
hist.history.keys()


plt.figure()
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.legend(['train_loss', 'val_loss'])
plt.show()

plt.figure()
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.legend(['train_acc', 'val_acc'])
plt.show()

#%%

results = model.evaluate(X_test,y_test)
print(results)

pred_y = np.argmax(model.predict(X_test),axis=1)
true_y = np.argmax(y_test,axis=1)

cm = confusion_matrix(true_y, pred_y)
cr = classification_report(true_y, pred_y)
print(cm) 
print(cr)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.show()






