import os
import random
from collections import Counter
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk import  WordNetLemmatizer
from nltk import NaiveBayesClassifier, classify
import re
from nltk.tokenize import RegexpTokenizer


sw=open('stopwords.txt')
stoplist=[line.strip().lower() for line in sw]
stoplist=stoplist+['yeah','I','We']
def loaddata(path):
	a_list=[]
	file_list=os.listdir(path)
	for file in file_list:
		#print("loading file"+file)
		f=open(path+file,'r')
		a_list.append(f.read())
	return a_list


def generatelist():
	ham=loaddata('enron3/ham/')
	print('ham emails:'+str(len(ham)))
	spam=loaddata('enron3/spam/')
	print('spam emails:'+str(len(spam)))
	spam_emails = [(email, 'spam') for email in spam]
	ham_emails = [(email, 'ham') for email in ham]
	all_emails = spam_emails + ham_emails
	random.shuffle(all_emails)
	return all_emails

def preprocess(sentence):
    lemmatizer = WordNetLemmatizer()
    tokenizer=RegexpTokenizer(r'\w+')
    #print(sentence)
    tokens=[]
    
    #using lemmatizer
    for word in tokenizer.tokenize(sentence.decode('ISO-8859-1')) :
    	if  not hasNumbers(word):
    		tokens.append(lemmatizer.lemmatize(word))

    #using stemmer
    #p_stem=PorterStemmer()
    #for word in tokenizer.tokenize(sentence.decode('ISO-8859-1')) :
    #	if  not hasNumbers(word):
    #		tokens.append(p_stem.stem(word))
    return tokens


def get_features(text, setting):
    if setting=='bow':
        return {word: count for word, count in Counter(preprocess(text)).items() if not word in stoplist}
    else:
        return {word: True for word in preprocess(text) if not word in stoplist}

def hasNumbers(inputstring):
	return any(char.isdigit() for char in inputstring)

def train(features, samples_proportion):
    train_size = int(len(features) * samples_proportion)
    # initialise the training and test sets
    train_set, test_set = features[:train_size], features[train_size:]
    print ('Training set size = ' + str(len(train_set)) + ' emails')
    print ('Test set size = ' + str(len(test_set)) + ' emails')
    # train the classifier
    classifier = NaiveBayesClassifier.train(train_set)
    return train_set, test_set, classifier

def evaluate(train_set, test_set, classifier):
    # check how the classifier performs on the training and test sets
    print ('Accuracy on the training set = ' + str(classify.accuracy(classifier, train_set)))
    print ('Accuracy of the test set = ' + str(classify.accuracy(classifier, test_set)))
    # check which words are most informative for the classifier
    classifier.show_most_informative_features(20)



if __name__ =="__main__":
    all_emails=generatelist()
    print ('Corpus size = ' + str(len(all_emails)) + ' emails')

    print('extracting features...')
    all_features = [(get_features(email, 'bow'), label) for (email, label) in all_emails]
    print ('Collected ' + str(len(all_features)) + ' feature sets')
    print(all_features)

    train_set, test_set, classifier = train(all_features, 0.8)

    evaluate(train_set, test_set, classifier)





