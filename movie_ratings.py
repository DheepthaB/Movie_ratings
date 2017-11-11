import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from collections import Counter

train=pd.read_csv("labeledTrainData.tsv",header=0,delimiter="\t",quoting=3)
positive_review=[]
negtive_review=[]
for i in range(0,15000):
	example1=BeautifulSoup(train["review"][i],"html.parser")
	example1_lower=example1.get_text().lower() 
	if train["sentiment"][i]==0:
		negtive_review.append(example1_lower)
	else:
		positive_review.append(example1_lower) 

def count_words(reviews):
	final_words=""
	for i in range(0,len(reviews)):
		letters_only=re.sub("[^A-Za-z]"," ",reviews[i])
		words=letters_only.split()
		words=[w for w in words if (w not in stopwords.words("english"))]
		for word in words:
			final_words+=word+" "
	return Counter(final_words)

positive_review_count=count_words(positive_review)
negative_review_count=count_words(negtive_review)
number_positive_review=len(positive_review)
number_negative_review=len(negtive_review)

prob_positive=number_positive_review/len(train["review"])
prob_negative=number_negative_review/len(train["review"])

#implements naive bayesian model with add-1 smoothing
def sentiment_prediction(text,review_count,class_prob,class_count):
	prediction=1
	words=Counter(text.split())
	for word in words:
		prediction*=words.get(word)*((review_count.get(word,0)+1)/(sum(review_count.values())+class_count))
	return prediction*class_prob


result=[]
test=pd.read_csv("testData.tsv",header=0,delimiter="\t",quoting=3);
for i in range(0,15000):
	test_example=BeautifulSoup(test["review"][i],"html.parser")
	test_example_lower=test_example.get_text().lower()
	negative_prediction = sentiment_prediction(test_example_lower,negative_review_count,prob_negative,number_negative_review)
	positive_prediction = sentiment_prediction(test_example_lower,positive_review_count,prob_positive,number_positive_review)
	if(positive_prediction>=negative_prediction):
		result[i]=1
	else:
		result[i]=0

output = pd.DataFrame(data={"id":test["id"][0:15000], "sentiment":result})
output.to_csv("Bag_of_Words_model.csv",index=False,quoting=3)

