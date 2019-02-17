# Remedy Classifier
# KMeans is use to cluser the Remedy dump and manually label the classificaiton based on the nature of the issue
# Use the labeled file to create the model.
# Use the model to classify the new issue
import pandas as pd
# with open("C:\Temp\RemedyData\RemedyTrain.dat","r") as text_file:
#     lines = text_file.read().split('\n')
path = "C:\Temp\RemedyData\RemedyTrain6.dat"
features = ['label','message'] 
sms = pd.read_table(path,header=None, names=features, delimiter='~') 
print('-------------Initilizing Train data-----------------')
# print(sms.head())
print(sms.label.value_counts())

# convert label to a numerical variable
sms['label_num'] = sms.label.map({
'Access_PHI_CHIA_PECA_Vault':1,
'Ticket Management':2,
'Support Group Management':3,
'Bulk Update':4,
'CI_BusinessService':5,
'Product Related':6,
'Email':7,
'Company_Vendor_Related':8,
'Load Issue':9,
'Reporting':10,
'Domain_Site':11,
'Ask_Template':12,
'CMS issue general remedy':13
})
#print(sms)

# how to define X and y (from the SMS data) for use with COUNTVECTORIZER
X = sms.message
y = sms.label_num
print(X.shape)
print(y.shape)

# 2. instantiate the vectorizer
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
# learn training data vocabulary, then use it to create a document-term matrix

# 3. fit
vect.fit(X)
#print(vect.get_feature_names())
# # 4. transform training data
# # equivalently: combine fit and transform into a single step
# # this is faster and what most people would do
X_train_dtm = vect.fit_transform(X)
# print('--------------')
# print(X_train_dtm)
print('-------------Model Created Successfuly----------------')
# ----------------Test Data ------------------------
path = "C:\Temp\RemedyData\RemedyTestData2018Q2Q3.dat"
features = ['message'] 
tst = pd.read_table(path,header=None, names=features) 
print('-------------Reading Test data--------------------------')
print(tst.head())
# how to define X and y (from the SMS data) for use with COUNTVECTORIZER
X1 = tst.message
# 4.1 transform testing data (using fitted vocabulary) into a document-term matrix
print('-------------Transforming Test data--------------------------')
X_test_dtm = vect.transform(X1)


# Building and evaluating a model

from sklearn.naive_bayes import MultinomialNB
# 2. instantiate a Multinomial Naive Bayes model
nb = MultinomialNB()
# 3. train the model 
nb.fit(X_train_dtm, y)

# 4. make class predictions for X_test_dtm
print('-------------Predicting Test data--------------------------')
y_pred_class = nb.predict(X_test_dtm)

# print(y_pred_class)

tst.insert(1,"Classification",y_pred_class)

tst['label_Class'] = tst.Classification.map({
1:'Access_PHI_CHIA_PECA_Vault',
2:'Ticket Management',
3:'Support Group Management',
4:'Bulk Update',
5:'CI_BusinessService',
6:'Product Related',
7:'Email',
8:'Company_Vendor_Related',
9:'Load Issue',
10:'Reporting',
11:'Domain_Site',
12:'Ask_Template',
13:'CMS issue general remedy'
})

# print(tst.head())
print('-------------Writing Result to File--------------------------')
tst.to_csv("C:\Temp\RemedyData\RemedyClassified2018Q2Q3V1.csv")
