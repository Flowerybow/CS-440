# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Last Modified 8/23/2023


"""
This is the main code for this MP.
You only need (and should) modify code within this file.
Original staff versions of all other files will be used by the autograder
so be careful to not modify anything else.
"""


import reader
import math
from tqdm import tqdm
from collections import Counter


'''
util for printing values
'''
def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")

"""
load_data loads the input data by calling the provided utility.
You can adjust default values for stemming and lowercase, when we haven't passed in specific values,
to potentially improve performance.
"""
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming: {stemming}")
    print(f"Lowercase: {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels


"""
Main function for training and predicting with naive bayes.
    You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""



def naive_bayes(train_set, train_labels, dev_set, laplace=1.0, pos_prior=0.5, silently=False):
    print_values(laplace,pos_prior)

    positive_counts = Counter()
    negative_counts = Counter()

    for review_as_list, label in zip(train_set, train_labels):
        if label == 1:
            positive_counts.update(review_as_list)
        else:
            negative_counts.update(review_as_list)

    positive_words = positive_counts.keys()
    negative_words = negative_counts.keys()

    vocabulary = set(positive_words) | set(negative_words)
    vocabulary_size = len(vocabulary)

    total_positive_words = positive_counts.total()
    total_negative_words = negative_counts.total()

    yhats = []

    pos_denominator = math.log(total_positive_words + laplace * vocabulary_size)
    neg_denominator = math.log(total_negative_words + laplace * vocabulary_size)

    for doc in dev_set:
        log_prob_positive = math.log(pos_prior)
        log_prob_negative = math.log(1 - pos_prior)
        for word in doc:
            pos_word_count = positive_counts.get(word, 0)
            log_prob_positive += math.log(pos_word_count + laplace)
            log_prob_positive -= pos_denominator
            
            neg_word_count = negative_counts.get(word,0)
            log_prob_negative += math.log(neg_word_count + laplace)
            log_prob_negative -= neg_denominator
        
        if log_prob_positive > log_prob_negative:
            yhats.append(1)
        else:
            yhats.append(0)

    return yhats
