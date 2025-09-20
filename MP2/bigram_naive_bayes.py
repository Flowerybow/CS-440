# bigram_naive_bayes.py
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
utils for printing values
'''
def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")

def print_values_bigram(unigram_laplace, bigram_laplace, bigram_lambda, pos_prior):
    print(f"Unigram Laplace: {unigram_laplace}")
    print(f"Bigram Laplace: {bigram_laplace}")
    print(f"Bigram Lambda: {bigram_lambda}")
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
Main function for training and predicting with the bigram mixture model.
    You can modify the default values for the Laplace smoothing parameters, model-mixture lambda parameter, and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""
def bigram_bayes(train_set, train_labels, dev_set, unigram_laplace=1.0, bigram_laplace=1.0, bigram_lambda=0, pos_prior=0.5, silently=False):
    print_values_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior)

    unigram_positive_counts = Counter()
    unigram_negative_counts = Counter()
    bigram_positive_counts = Counter()
    bigram_negative_counts = Counter()

    for review_as_list, label in zip(train_set, train_labels):
        bigrams_in_review = []
        for i in range(len(review_as_list) - 1):
            bigram = (review_as_list[i], review_as_list[i+1])
            bigrams_in_review.append(bigram)
        if label == 1:
            bigram_positive_counts.update(bigrams_in_review)
            unigram_positive_counts.update(review_as_list)
        else:
            bigram_negative_counts.update(bigrams_in_review)
            unigram_negative_counts.update(review_as_list)

    total_positive_unigrams = unigram_positive_counts.total()
    total_negative_unigrams = unigram_negative_counts.total()
    total_positive_bigrams = bigram_positive_counts.total()
    total_negative_bigrams = bigram_negative_counts.total()

    unigram_positive_words = unigram_positive_counts.keys()
    unigram_negative_words = unigram_negative_counts.keys()

    unigram_vocabulary = set(unigram_positive_words) | set(unigram_negative_words)
    unigram_vocabulary_size = len(unigram_vocabulary)


    yhats = []
    for doc in dev_set:
        log_prob_positive = math.log(pos_prior)
        log_prob_negative = math.log(1 - pos_prior)
        if len(doc) > 0:
            word = doc[0]
            p_uni_pos = (unigram_positive_counts.get(word, 0) + unigram_laplace) / (total_positive_unigrams + unigram_laplace * unigram_vocabulary_size)
            log_prob_positive += math.log(p_uni_pos)

            p_uni_neg = (unigram_negative_counts.get(word, 0) + unigram_laplace) / (total_negative_unigrams + unigram_laplace * unigram_vocabulary_size)
            log_prob_negative += math.log(p_uni_neg)

        for i in range(1, len(doc)):
            unigram = doc[i]
            bigram = (doc[i-1], doc[i])
            p_uni_pos = (unigram_positive_counts.get(unigram, 0) + unigram_laplace) / (total_positive_unigrams + unigram_laplace * unigram_vocabulary_size)
            p_bi_pos = (bigram_positive_counts.get(bigram, 0) + bigram_laplace) / (total_positive_bigrams + bigram_laplace *unigram_vocabulary_size)

            mixture_pos = (1 - bigram_lambda) * p_uni_pos + bigram_lambda * p_bi_pos
            log_prob_positive += math.log(mixture_pos)

            p_uni_neg = (unigram_negative_counts.get(unigram, 0) + unigram_laplace) / (total_negative_unigrams + unigram_laplace * unigram_vocabulary_size)
            p_bi_neg = (bigram_negative_counts.get(bigram, 0) + bigram_laplace) / (total_negative_bigrams + bigram_laplace * unigram_vocabulary_size)

            mixture_neg = (1 - bigram_lambda) * p_uni_neg + bigram_lambda * p_bi_neg
            log_prob_negative += math.log(mixture_neg)

        if log_prob_positive > log_prob_negative:
            yhats.append(1)
        else:
            yhats.append(0)

    return yhats



