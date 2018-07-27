import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier
from multiprocessing import Process, Queue
import models
import torch.nn as nn
import torch
import torch.optim as optim
import os
import metric_learn
from run4tests import kNN, readInputCSV
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def main():
    modelName = input("Model name: ")
    dataset = input("Dataset: ")
    authorList, Xtrain, Ytrain, Xtest, Ytest = readInputCSV(modelName, dataset)
    print("Data successfully loaded.")

    pca = LinearDiscriminantAnalysis(n_components=2)
    X_r = pca.fit(Xtrain, Ytrain).transform(Xtrain)


    colors = ['navy', 'turquoise', 'darkorange', 'blue', 'green', 'red', 'yellow', 'purple', 'brown', 'black']
    lw = 2

    plt.figure()
    for stidx in range(0,500,50):
        authoridx = int(stidx / 50)
        plt.scatter(X_r[stidx:stidx+50, 0], X_r[stidx:stidx+50, 1], color=colors[authoridx], alpha=.8, lw=lw,label=str(authoridx))
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('LDA of CCAT10 word model - train set')
    plt.savefig("trainset.png")

    pca = PCA(n_components=2)
    X_r = pca.fit(Xtest).transform(Xtest)

    plt.figure()
    for stidx in range(0, 500, 50):
        authoridx = int(stidx / 50)
        plt.scatter(X_r[stidx:stidx + 50, 0], X_r[stidx:stidx + 50, 1], color=colors[authoridx], alpha=.8, lw=lw,
                    label=str(authoridx))
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('LDA of CCAT10 word model - test set')
    plt.savefig("testset.png")

if __name__ == "__main__":
    main()