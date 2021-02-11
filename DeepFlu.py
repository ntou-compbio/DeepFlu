# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 21:24:56 2020

@author: anna zan
"""

import pandas as pd
from sklearn import preprocessing
import sys

from keras.models import Sequential
from keras.layers import Dense,Dropout

# read probes
import all_22277_probelist

all_df = pd.read_csv("data/t0_data/H1N1_rmat0_001_t0_no237/H1N1_rmat0_001_t0_no237.txt",sep='\t',encoding='utf-8') #read H1N1_rmat0_001_t0_no237file

cols=all_22277_list.cols #import prrobes
all_df=all_df[cols]

def PreprocessData(raw_df): #format data
   
    df=raw_df.drop(['ID'], axis=1) #remove ID
    ndarray = df.values
    Features = ndarray[:,1:]#extract features
    Label = ndarray[:,0]#extract label

    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1)) 
    scaledFeatures=minmax_scale.fit_transform(Features)    
    
    return scaledFeatures,Label

all_Features,all_Label=PreprocessData(all_df)
test_Features=all_Features[:2]#extract features for test data (two time points)
test_Label=all_Label[:2]#extract labels for test data (two time points)
train_Features=all_Features[2:]#extract features for training data
train_Label=all_Label[2:]#extract labels for training data

#build the model
#the model has one input layer with 22277 nodes, four hidden layers with 100 nodes each (first hidden layer with 0.1 dropout), one output layer with one node
model = Sequential()
model.add(Dense(units=100, input_dim=22277, kernel_initializer='uniform', activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=100, kernel_initializer='uniform', activation='relu'))
model.add(Dense(units=100, kernel_initializer='uniform', activation='relu'))
model.add(Dense(units=100, kernel_initializer='uniform', activation='relu'))
model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
train_history =model.fit(x=train_Features, y=train_Label, validation_split=0.1, epochs=150, batch_size=200,verbose=0)
scores=model.evaluate(x=test_Features, y=test_Label)

# test the model
all_Features,Label=PreprocessData(all_df)
all_probability=model.predict_classes(all_Features)
all_probability_score=model.predict_proba(all_Features)
pd=all_df
pd.insert(len(all_df.columns), 'probability',all_probability)
predict=all_probability[:2]#obtain predicted labels for test data
predict_score=all_probability_score[:2]#obtain prediction probability for test data 
predict=predict.tolist()
predict_score=predict_score.tolist()

# show the performance metrics
f1 = open("./H1N1_t0_001.txt", 'a', encoding = 'UTF-8')#customize the file name to store the performance result
f1.write(str(int(test_Label[0]))+"\t"+str(predict[0])+"\t"+str(predict[1])+"\t"+str(predict_score[0])+"\t"+str(predict_score[1])+"\n")#print test label / predicted label (two time points)/ prediction probability (two time points)
f1.close