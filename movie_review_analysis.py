import nltk
from nltk.corpus import movie_reviews
import random
import re
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import BernoulliNB

# Store data into a dictionary of documents
documents = [(list(movie_reviews.words(fileid)), category)
              for category in movie_reviews.categories()
              for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

# Creating a frequency distribution for all words
all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())
    
all_words = nltk.FreqDist(all_words)    

# Define the feature extractor with tokenization and stop words removal
word_features = list(all_words)[:2000]

def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features
    
# Train using Naive Bayes classifier
feature_sets = [(document_features(d), c) for (d,c) in documents]
train_set, test_set = feature_sets[200:], feature_sets[:200]
classifier = nltk.NaiveBayesClassifier.train(train_set)

# Show classifier accuracy and 10 most informative features
print(nltk.classify.accuracy(classifier, test_set))
classifier.show_most_informative_features(10)

# Train using Bernoulli classifier
classifier = SklearnClassifier(BernoulliNB()).train(train_set)
print(nltk.classify.accuracy(classifier, test_set))
