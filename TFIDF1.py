
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 08:37:07 2023

@author: Admin
"""

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
corpus=['The mouse had tiny litle mouse','The cat saw the mouse','The cat catch the mouse','The end of the mouse story']
#Step 1 initalize the count vector
cv=CountVectorizer()
#To count the total number of Tf
word_count_vector=cv.fit_transform(corpus)
word_count_vector.shape
#Now the next step is to apply IDF
tfidf_trnasformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_trnasformer.fit(word_count_vector)
#This matrix is in raw matrix form ,let us convert it into dataframe
df_idf=pd.DataFrame(tfidf_trnasformer.idf_,index=cv.get_feature_names_out(),columns=['idf_weights'])
#sort ascending
df_idf.sort_values(by=['idf_weights'])




#How to find TFIDF actually to the dataset
from sklearn.feature_extraction.text import TfidfVectorizer
corpus=[
        "Thor eating pizza,Loki is eating pizza,Ironman ate pizaa already",
        "Apple is aanouncing new iphone tomorrow",
        "Tesla is announcing new model-3 tomorrow",
        "Google is aanouncing new pixel-6 tomorrow",
        "Microsoft is aanouncing new surface tomorrow",
        "Amazon is aanouncing new eco-dot tomorrow",
        "T am eating biryani and you are eating grapes"]

#Let us create the vectorizer and fit the corpus and transform them according
v=TfidfVectorizer()
v.fit(corpus)
transform_output=v.transform(corpus)


#Let us print vocabulory
print(v.vocabulary_)


#let us print the idf of each word:
all_feature_names=v.get_feature_names_out()

    #Lets get index in the vocabulory
    
    
    



############################################################################
###################Actual performance of TFIDf with the dataset
import pandas as pd
#Read the data into the pandas dataframe
df=pd.read_csv("D:/Data Science/6-Datasets/Ecommerce_data.csv.xls")
print(df.shape)
df.head(5)
#Check the distribution of the labels
df['label'].value_counts()
#Add the new col which gives a unique number ti each of these labels
df['Label_num']=df['label'].map({
    'Household':0,
    'Books':1,
    'Electronics':2,
    'Clothing & Accessories':3})


#Checking the result
df.head(5)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(
    df.Text,
    df.Label_num,
    test_size=0.2,
    random_state=2022,
    stratify=df.Label_num)

print("shape of X_train:",x_train.shape)
print("shape of x_test:",x_test.shape)
y_train.value_counts()
y_test.value_counts()

##################################################
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


#1.Create a pipeline object
clf=Pipeline([
    ('vectorizer_tfidf',TfidfVectorizer()),
    ('KNN',KNeighborsClassifier())])

#2.Fit with x_trin and y_train
clf.fit(x_train,y_train)


#3.Get the prediction for x_test and store it in y_pred
y_pred=clf.predict(x_test)


#4.Print the classification report
print(classification_report(y_test, y_pred))
#).96 means ovefitiing 
