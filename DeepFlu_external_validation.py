# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 12:45:13 2020

@author: anna zan
"""

import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn import preprocessing, metrics

from keras.models import Sequential
from keras.layers import Dense,Dropout

import all_12023_probelist

# read H1N1 external dataset
all_df = pd.read_csv("data/external_validation_data/73072_H1N1_external_validation_D3_D4.txt",sep='\t',encoding='utf-8')
# read H3N2 external dataset
# all_df = pd.read_csv("data/external_validation_data/73072_H3N2_external_validation_D2_D5.txt",sep='\t',encoding='utf-8')#讀取73072_H3N2_external_validation_D2_D5.txt檔案

cols=all_12023_list.cols#import prrobes
all_df=all_df[cols]

def PreprocessData(raw_df):#format data
   
    df=raw_df.drop(['ID'], axis=1)#remove ID
    ndarray = df.values
    Features = ndarray[:,1:]#extract features
    Label = ndarray[:,0]#extract label

    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaledFeatures=minmax_scale.fit_transform(Features)    
    
    return scaledFeatures,Label

all_Features,all_Label=PreprocessData(all_df)
train_Features=all_Features[:40]#extract features for training data
train_Label=all_Label[:40]#extract labels for training data
test_Features=all_Features[38:]#extract features for test data
test_Label=all_Label[38:]#extract labels for test data

#build the model
#the model has one input layer with 12023 nodes, four hidden layers with 100 nodes each (first hidden layer with 0.1 dropout), one output layer with one node
model = Sequential()
model.add(Dense(units=100, input_dim=12023, kernel_initializer='uniform', activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=100, kernel_initializer='uniform', activation='relu'))
model.add(Dense(units=100, kernel_initializer='uniform', activation='relu'))
model.add(Dense(units=100, kernel_initializer='uniform', activation='relu'))
model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
train_history =model.fit(x=train_Features, y=train_Label, validation_split=0.1, epochs=150, batch_size=200,verbose=0)

#test the model
all_Features,Label=PreprocessData(all_df)
all_probability=model.predict_classes(all_Features)
all_probability_score=model.predict_proba(all_Features)
pd=all_df
pd.insert(len(all_df.columns), 'probability',all_probability)
predict=all_probability[38:]#obtain predicted labels for test data
predict_score=all_probability_score[38:]#obtain prediction probability for test data 
predict_score=predict_score.tolist()

# show performance metrics
fpr, tpr, thresholds = metrics.roc_curve(test_Label, predict)
auc_roc = metrics.auc(fpr, tpr) #compute AUROC

precision, recall, thresholds = precision_recall_curve(test_Label, predict)
auc_pr = metrics.auc(recall, precision) #compute AUPR

TP=0
TN=0
FP=0
FN=0
for j in range(predict.size):
    if(predict[j]==1 and predict[j]==test_Label[j]):
        TP=TP+1
    else:
        TP=TP+0
    
    if(predict[j]==0 and predict[j]==test_Label[j]):
        TN=TN+1
    else:
        TN=TN+0
    
    if(test_Label[j]==0 and predict[j]==1):
        FP=FP+1
    else:
        FP=FP+0
    
    if(test_Label[j]==1 and predict[j]==0):
        FN=FN+1
    else:
        FN=FN+0

acc=(TP+TN)/(TP+TN+FP+FN) #Accuracy
sen=TP/(TP+FN) #Sensitivity
spe=TN/(TN+FP) #Specificity
pre=TP/(TP+FP) #Precision
    
print(str(acc)+"\t"+str(sen)+"\t"+str(spe)+"\t"+str(pre)+"\t"+str(auc_roc)+"\t"+str(auc_pr)+"\n")#列印出各指標數值