# Remedy Classifier
# KMeans is use to cluser the Remedy dump and manually label the classificaiton based on the nature of the issue
# Use the labeled file to create the model.
# Use the model to classify the new issue
import pandas as pd

# with open("C:\Temp\RemedyData\RemedyTrainInter.dat","r") as text_file:
#     lines = text_file.read().split('\n')
# C:\Temp\RemedyData\RemedyTrain2.dat

path = "C:\Temp\RemedyData\RemedyTrain6.dat"
features = ['label','message'] 
sms = pd.read_table(path,header=None, names=features, delimiter='~') 

# print(sms.head())
# print(sms.label.value_counts())
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
'CMS issue general remedy':13,
'KBA':14,
'QuerySearch_Console':15
})
# print(sms)

# how to define X and y (from the SMS data) for use with COUNTVECTORIZER
X = sms.message
y = sms.label_num
# print(X.shape)
# print(y.shape)

# split X and y into training and testing sets
# by default, it splits 75% training and 25% test
# random_state=1 for reproducibility
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1,test_size=0.25)
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)

# 2. instantiate the vectorizer
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
# learn training data vocabulary, then use it to create a document-term matrix

# 3. fit
vect.fit(X_train)
# print(vect.get_feature_names())
# # 4. transform training data
# # equivalently: combine fit and transform into a single step
# # this is faster and what most people would do
X_train_dtm = vect.fit_transform(X_train)
# print('--------------')
# print(X_train_dtm)
# 4.1 transform testing data (using fitted vocabulary) into a document-term matrix
X_test_dtm = vect.transform(X_test)

# Building and evaluating a model

from sklearn.naive_bayes import MultinomialNB
# 2. instantiate a Multinomial Naive Bayes model
nb = MultinomialNB()
# 3. train the model 
nb.fit(X_train_dtm, y_train)
# 4. make class predictions for X_test_dtm
y_pred_class = nb.predict(X_test_dtm)
# calculate accuracy of class predictions
from sklearn import metrics
print('Predict Accuracy : ')
print(metrics.accuracy_score(y_test, y_pred_class))

# examine class distribution
# print(y_test.value_counts())
# there is a majority class of 0 here, hence the classes are skewed

# calculate null accuracy (for multi-class classification problems)
# .head(1) assesses the value 1208
null_accuracy = y_test.value_counts().head(1) / len(y_test)
# print('Null accuracy:', null_accuracy)

# Manual calculation of null accuracy by always predicting the majority class
# print('Manual null accuracy:',(1196 / (1196 + 1+306+201+22+8+4)))

# print the confusion matrix
# print(metrics.confusion_matrix(y_test, y_pred_class))



# 4. make class predictions for X_test_dtm
print('----------Testing---------')
print(nb.predict(vect.transform(['Type of Request: Im seeing an issue  ***If Support Issue or Need Work Done*** Description: UTN 422814194 CRQ CRQ000002492311  Ticket needs to be cancelled, but   to do so. get weird message "You do not have   to move to the status of . (ARERR 44856)"  ***If Access*** Who Is the Access For?:  Login ID:  Users Name:  Support Organization:  -  Change Management Access Required:  Has proof been attached:  Requested Support Groups:  Non-Prod Domain Access:  Additional Details: '])))
