# Kieran Parikh
# Script to load a trained encoder and run a classifier on it.
# UCCS REU - June 2018 
import numpy as np
import models
import pandas as pd
import csv

import torch
from torch.autograd import Variable
import torch.nn as nn

import os
import sys
import time
import argparse

def main():
    datasetName = input("Dataset name: ")
    modelName = input("Model name: ")
    embeddingType = input("Embedding Type (ptngrams, lngrams, lwords, ptwords) : ")
    torch.cuda.set_device(2)
    modelFile = datasetName+'-'+modelName+'.pickle'
    encoder = loadEncoder(modelFile, embeddingType)
    global stop_words
    # train data
    with open('encoded/'+datasetName+'-encodedTrainData-'+modelName+'.csv', 'w+', newline='') as outfile:
        writer = csv.writer(outfile,  dialect='excel')
        dataset = readInputCSV('train', datasetName)
        encodeDataset(dataset, encoder, writer)
    # test data
    with open('encoded/'+datasetName+'-encodedTestData-'+modelName+'.csv', 'w+', newline='') as outfile:
        writer = csv.writer(outfile,  dialect='excel')
        dataset = readInputCSV('test', datasetName)
        encodeDataset(dataset, encoder, writer)

    return

def loadEncoder(modelName, embeddingType):
    savedir = 'savedir/'
    model = torch.load(os.path.join(savedir, modelName))
    encoder = model.encoder
    if embeddingType == 'ptngrams':
        encoder.set_glove_path("dataset/GloVe/charNgram.txt")
    elif embeddingType == 'lngrams':
        encoder.set_glove_path("dataset/GloVe/C50-traintest-3grams.txt")
    elif embeddingType == 'lwords':
        encoder.set_glove_path("dataset/GloVe/traintestwordmodel.txt")
    else:
        encoder.set_glove_path("dataset/GloVe/glove.840B.300d.txt")
    return encoder

def readInputCSV(dataType, datasetName):
    with open('dataset/'+datasetName+'-'+dataType+'Data.csv', newline='', encoding="utf-8") as csvfile:
        data = pd.read_csv(csvfile, delimiter=',', names = ['text', 'author'], encoding='utf-8')
    return data

def encodeDataset(dataset, encoder, writer):
    texts = dataset['text'].tolist()
    encoder.build_vocab(texts, False)

    embedding = encoder.encode(texts)
    for index, row in dataset.iterrows():
            writeRow = embedding[index].tolist()
            writeRow.append(row['author'])
            writer.writerow(writeRow)
    return

if __name__ == "__main__":
    main()
