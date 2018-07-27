# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import numpy as np
import torch
import pandas as pd
import re
from nltk import ngrams

def splitSent(text, splitType='word', n=3):
    if splitType=='n-grams':
        out_list = list()
        for ngram in ngrams(list(text), n):
            out_list.append(''.join(ngram))
        return out_list
    else:
        return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

def get_batch(batch, word_vec, word_dim):
    # sent in batch in decreasing order of lengths (bsize, max_len, word_dim)
    lengths = np.array([len(x) for x in batch])
    max_len = np.max(lengths)
    embed = np.zeros((max_len, len(batch), word_dim))

    for i in range(len(batch)):
        for j in range(len(batch[i])):
            embed[j, i, :] = word_vec[batch[i][j]]

    return torch.from_numpy(embed).float(), lengths


def get_word_dict(sentences, splitType='word'):
    # create vocab of words
    word_dict = {}
    for sent in sentences:
        for word in splitSent(sent, splitType=splitType):
            if word not in word_dict:
                word_dict[word] = ''

    '''
    word_dict['<s>'] = ''
    word_dict['</s>'] = ''
    word_dict['<p>'] = ''
    '''
    return word_dict


def get_glove(word_dict, glove_path, splitType='word'):
    # create word_vec with glove vectors
    word_vec = {}
    with open(glove_path, encoding="utf-8") as f:
        for line in f:

            if 'charNgram.txt' in glove_path:
                type, line = line.split('-', 1)

            word, vec = line.split(' ', 1)
            # fix embeddings
            word = word.replace('~', ' ')
            word = word.replace('*n', '\n')
            if word in word_dict:
                word_vec[word] = np.array(list(map(float, vec.split())))
    print('Found {0}(/{1}) words with glove vectors'.format(
                len(word_vec), len(word_dict)))
    return word_vec


def build_vocab(sentences, glove_path, splitType='word'):
    word_dict = get_word_dict(sentences, splitType=splitType)
    word_vec = get_glove(word_dict, glove_path, splitType=splitType)
    print('Vocab size : {0}'.format(len(word_vec)))
    return word_vec


def get_nli(data_path, datasetName='C50'):
    s1 = {}
    s2 = {}
    target = {}

    dico_label = {'not same author': 0,  'same author': 1}

    # Load train data
    filepath = os.path.join(data_path, datasetName+'-trainData.csv')
    with open(filepath, newline='', encoding="utf-8") as csvfile:
        data = pd.read_csv(csvfile, delimiter=',', names=['text', 'author'],  encoding='utf-8')
        authorList = list()
        currAuthor = 0
        currList = list()
        for index, row in data.iterrows():
            # Check if current entry is of same author
            if row['author'] != currAuthor:
                #If current entry is of a different author, reset
                authorList.append(currList)
                currList = list()
                currAuthor = row['author']
            # Attribute current row to current author's list
            currList.append(row['text'])
        authorList.append(currList)
        texts = data['text'].tolist()
        numTexts = len(texts)        
        print('** TRAIN DATA : Found {0} texts.'.format(numTexts))


    for data_type in ['dev', 'test']:
        s1[data_type], s2[data_type], target[data_type] = {}, {}, {}
        filepath = os.path.join(data_path, datasetName+'-encoderTrainData-' + data_type + '.csv')
        with open(filepath, newline='', encoding="utf-8") as csvfile:
            data = pd.read_csv(csvfile, delimiter=',', names=['s1', 's2', 'target'], encoding='utf-8')
            s1[data_type]['sent'] = data['s1'].tolist()
            s2[data_type]['sent'] = data['s2'].tolist()
            #target[data_type]['data'] = np.array(data['target'].tolist()).astype(int)

            targetList = list()
            for entry in data['target'].tolist():
                if entry:
                    targetList.append([1])
                else:
                    targetList.append([0])
            target[data_type]['data'] = np.array(targetList)
                        
        assert len(s1[data_type]['sent']) == len(s2[data_type]['sent']) == len(target[data_type]['data'])

        print('** {0} DATA : Found {1} pairs of {2} texts.'.format(
                data_type.upper(), len(s1[data_type]['sent']), data_type))

    train = {'authorList': authorList, 'numTexts': numTexts, 'texts': texts}
    dev = {'s1': s1['dev']['sent'], 's2': s2['dev']['sent'],'label': target['dev']['data']}
    test = {'s1': s1['test']['sent'], 's2': s2['test']['sent'],'label': target['test']['data']}
    return train, dev, test
