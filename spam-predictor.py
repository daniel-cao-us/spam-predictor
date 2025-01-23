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
"""
def load_data(trainingdir, testdir, stemming=False, lowercase=True, silently=False):
    print(f"Stemming: {stemming}")
    print(f"Lowercase: {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels


"""
Main function for training and predicting with naive bayes.
"""
def naive_bayes(train_set, train_labels, dev_set, laplace=1, pos_prior=0.8, silently=False):
    
    #training phase, find probability that each word P(word, given that it is positive), P(word, given that it is negative)
    pos_word_counts = Counter()
    neg_word_counts = Counter()

    total_pos_words = 0
    total_neg_words = 0

    #keep dict of words and their counts for each type: negative, positive
    for i in range(len(train_set)):

        if train_labels[i] == 1: #then the review is positive
            pos_word_counts.update(train_set[i]) #add to counts of each words
            total_pos_words += len(train_set[i]) #keep total count of positive words
        elif train_labels[i] == 0: #then the review is negative
            neg_word_counts.update(train_set[i]) #add to counts of each negative words
            total_neg_words += len(train_set[i]) #keep total count of negative words
    
    #count number of total unique words
    total_unique_words = set()
    for review in train_set:
        for word in review:
            total_unique_words.add(word)
            
    num_unique_words = len(total_unique_words)

    #calculate probability of each word using laplace smoothing
    prob_pos_words = {}
    prob_neg_words = {}

    #calculate prob of unseen words
    pos_unseen_prob = math.log((laplace)/(total_pos_words + laplace * (num_unique_words + 1)))
    neg_unseen_prob = math.log((laplace)/(total_neg_words + laplace * (num_unique_words + 1)))

    #calculate prob of already seen words
    for word, word_count in pos_word_counts.items():
        prob_pos_words[word] = math.log((pos_word_counts[word] + laplace)/(total_pos_words + laplace * (num_unique_words + 1)))
    for word, word_count in neg_word_counts.items():
        prob_neg_words[word] = math.log((neg_word_counts[word] + laplace)/(total_neg_words + laplace * (num_unique_words + 1)))
    
    #test on development data
    labels = []
    for doc in tqdm(dev_set, disable=silently):
        #add probabilities instead of multiply
        pos_prob_sum = 0
        neg_prob_sum = 0

        for word in doc:
            #calculate sum of P(Word | Positive)
            if word in prob_pos_words: #if word is in positive dataset
                pos_prob_sum += prob_pos_words[word]
            else: #if unseen word
                pos_prob_sum += pos_unseen_prob

            #calculate sum of P(Word | Negative)
            if word in prob_neg_words: #if word is in negative dataset
                neg_prob_sum += prob_neg_words[word]
            else: #if unseen word
                neg_prob_sum += neg_unseen_prob

        posterior_pos = math.log(pos_prior) + pos_prob_sum
        posterior_neg = math.log(1-pos_prior) + neg_prob_sum

        if (posterior_pos > posterior_neg):
            labels.append(1)
        else:
            labels.append(0)

    return labels
