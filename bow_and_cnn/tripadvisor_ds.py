from __future__ import print_function
from __future__ import division


import pandas as pd 
import numpy as np
import glob

from common import utils
from common import vocabulary
from nltk import word_tokenize
from importlib import reload

import os, sys, re, json, time, datetime, shutil

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle


reload(utils)
reload(vocabulary)

def isNaN(num):
    return num != num

def read_file(filepath):
    df = pd.DataFrame()
    for path in filepath:
        for file in glob.glob(path):
            print(file)
            temp_df = pd.read_csv(file)
            city = file.split('/')[-1]
            temp_df['city'] = '_'.join(city.split('_')[:-1])
            temp_df['review'] = temp_df.review_title + ' ' + temp_df.review_body
            temp_df.rating = temp_df.rating/10
            temp_df.rating = temp_df.rating.astype(int)
            df = df.append(temp_df)
            
    df = df.reset_index(drop=True)
    return df

def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    if iteration == total: 
        print(' ')
        

class TripAdvisor_DS(object):
    
    def __init__(self):
        
        self.master_df = None
        self.vocab = None
        
            

    def process(self, train_test_split = 0.9, train_validate_split = 0.9, shuffle = True, input_length=500, sample=False):

        if sample:
            self.filepath = ['../reviews/Chicago_Illinois_3.csv']
        else:
            self.filepath = ['../reviews/*.csv']
        self.master_df = read_file(self.filepath)
        
        
        self.input_length = input_length
        
        # shuffle 
        if shuffle:
            self.master_df = self.master_df.sample(frac=1)
            self.master_df = self.master_df.reset_index(drop=True)
            
        # split 
        train_test_split_count = int(self.master_df.shape[0] * train_test_split)
        train_validate_split_count = int(train_test_split_count * train_validate_split)
        

        # train, validate and test dataframe
        self.train_df = self.master_df.loc[:train_validate_split_count]
        self.validate_df = self.master_df.loc[train_validate_split_count:train_test_split_count]
        self.test_df = self.master_df.loc[train_test_split_count:]
        

        # build vocab over training set    
        tokens = word_tokenize(self.train_df.review.to_string().lower())
        print("Tokens: {}".format(len(set(tokens))))
        self.vocab = vocabulary.Vocabulary(set(utils.canonicalize_word(w) for w in tokens))
        print("Vocabulary: {:,} types".format(self.vocab.size))
       
    
        # build training, validation and test set 
        self.train_features = []
        self.train_labels = []
        self.validate_features = []
        self.validate_labels = []
        self.test_features = []
        self.test_labels = []
      
        for i in range(0, self.master_df.shape[0], 1): 

            if not isNaN(self.master_df.loc[i].review):
                
                tokens = word_tokenize(self.master_df.loc[i].review.lower())
                feature = self.vocab.words_to_ids(utils.canonicalize_word(w, self.vocab) for w in tokens)
                
                #rating = self.master_df.loc[i].rating - 1
                
                
                if (self.master_df.loc[i].rating == 1) or (self.master_df.loc[i].rating == 2):
                    rating = 0
                elif self.master_df.loc[i].rating == 3:
                    rating = 1
                elif self.master_df.loc[i].rating == 4:
                    rating = 2
                else:
                    rating = 3
                
                if i < train_validate_split_count:
                    self.train_features.append(feature)
                    self.train_labels.append(rating)
                else:
                    if i < train_test_split_count:
                        self.validate_features.append(feature)
                        self.validate_labels.append(rating)
                    else:
                        self.test_features.append(feature)
                        self.test_labels.append(rating)
            printProgressBar(i, self.master_df.shape[0]-1)            

        self.train_features = np.asarray(self.train_features)
        self.train_labels = np.asarray(self.train_labels)
        self.validate_features = np.asarray(self.validate_features)
        self.validate_labels = np.asarray(self.validate_labels)
        self.test_features = np.asarray(self.test_features)
        self.test_labels = np.asarray(self.test_labels)

        print("number of train_features = {}, train_labels = {}".format(len(self.train_features), len(self.train_labels)))
        print("number of validate_features = {}, validate_labels = {}".format(len(self.validate_features), len(self.validate_labels)))
        print("number of test_features = {}, test_labels = {}".format(len(self.test_features), len(self.test_labels)))
    
    
        self.padded_train_features, self.train_ns = utils.pad_np_array(self.train_features, self.input_length)
        self.padded_validate_features, self.validate_ns = utils.pad_np_array(self.validate_features, self.input_length)
        self.padded_test_features, self.test_ns = utils.pad_np_array(self.test_features, self.input_length)
        
        self.target_labels = [0, 1, 2, 3]
        
        return self

    def get_padded_ids(self, sent):
        
        tokens = word_tokenize(sent.lower())    
        ids = np.asarray([ds.vocab.words_to_ids(utils.canonicalize_word(w, ds.vocab) for w in tokens)])

        padded_ids, _ = utils.pad_np_array(ids, self.input_length)
        return padded_ids

    
    def save(self, data_file):
            
        max_bytes = 2**31 - 1
        bytes_out = pickle.dumps(ds)
        with open(data_file, 'wb') as f_out:
            for idx in range(0, len(bytes_out), max_bytes):
                f_out.write(bytes_out[idx:idx+max_bytes])


    def get_ord_labels(self):
        
                    
        self.train_ord_labels = [[1]*w + [0]*(len(self.target_labels)-w - 1) for w in self.train_labels]
        self.validate_ord_labels = [[1]*w + [0]*(len(self.target_labels)-w - 1) for w in self.validate_labels]
        self.test_ord_labels = [[1]*w + [0]*(len(self.target_labels)-w - 1) for w in self.test_labels]

                    
        
        
    