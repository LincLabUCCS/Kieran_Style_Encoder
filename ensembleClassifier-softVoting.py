import numpy as np
import pandas as pd
from sklearn import svm, tree, naive_bayes, linear_model
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from multiprocessing import Process, Queue
import models
import torch.nn as nn
import torch
import torch.optim as optim
import os
import pdb
import dill

def main():
    dataset = input("Dataset name: ")
    modelName = input("Model name: ")
    k = int(input("k value: "))
    global Ytest
    authorList, Xtrain, Ytrain, Xtest, Ytest = readInputCSV(modelName, dataset)
    print("Data successfully loaded.")

    clf1 = svm.SVC(decision_function_shape='ovr', probability=True, cache_size=4000)
    clf2 = KNeighborsClassifier(n_neighbors=5)
    clf3 = RandomForestClassifier(random_state=1)
    clf4 = LogisticRegression(random_state=1)

    voter = VotingClassifier(estimators=[('SVM', clf1), ('kNN', clf2), ('RandomForest', clf3), ('logistic', clf4)], voting='hard')
    voter.fit(Xtrain, Ytrain)

    train_acc = voter.score(Xtrain, Ytrain)
    test_acc = voter.score(Xtest, Ytest)

    print('train acc: ' + str(train_acc) + '; test acc: ' + str(test_acc))



def SVM(modelName, Xtrain, Ytrain, Xtest, Ytest, q, dataset, trainPredQ, testPredQ):
    mySVM = svm.SVC(decision_function_shape='ovo', cache_size=4000)
    mySVM.fit(Xtrain, Ytrain)
    print("SVM fit to training data. Beginning predictions.")

    joblib.dump(mySVM, 'savedir/SVMs/'+dataset+'-'+modelName+'.pickle')

    predictions = mySVM.predict(Xtrain)
    correct = 0.0
    for pred, label in zip(predictions, Ytrain):
        if pred == label:
            correct = correct + 1
    train_acc =100 * correct / len(Ytrain)
    trainPredQ.put({'svm': predictions})


    predictions = mySVM.predict(Xtest)
    correct = 0.0
    for pred, label in zip(predictions, Ytest):
        if pred == label:
            correct = correct + 1
    test_acc =100 * correct / len(Ytest)

    returnDict = {'svm_train_acc': train_acc, 'svm_test_acc': test_acc}
    q.put(returnDict)
    testPredQ.put({'svm': predictions})

def kNN(k, Xtrain, Ytrain, Xtest, Ytest, q, trainPredQ, testPredQ):
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(Xtrain, Ytrain)

    predictions = neigh.predict(Xtrain)
    correct = 0.0
    for pred, label in zip(predictions, Ytrain):
        if pred == label:
            correct = correct + 1
    train_acc = 100 * correct / len(Ytrain)
    trainPredQ.put({'knn': predictions})

    predictions = neigh.predict(Xtest)
    correct = 0.0
    for pred, label in zip(predictions, Ytest):
        if pred == label:
            correct = correct + 1
    test_acc = 100 * correct / len(Ytest)

    returnDict = {'knn_train_acc': train_acc, 'knn_test_acc': test_acc}
    q.put(returnDict)
    testPredQ.put({'knn': predictions})

def cohort(K, modelName, authorList, Xtrain, Ytrain, Xtest, Ytest, q, dataset, trainPredQ, testPredQ):
    classifier = loadClassifier(modelName, dataset)
    print("Classifier successfully loaded.")
    correct = 0.0
    sigmoid = nn.Sigmoid()
    pred = list()
    batch_size = 30

    for i_index in range(len(Xtrain)):
        i = Xtrain[i_index]
        votes = list()

        for author in authorList:
            currAuthorVotes = 0
            texts = author[:K]
            for stidx in range(0, K, batch_size):
                u = torch.tensor(np.tile(i,(batch_size,1))).float().cuda()
                v = torch.tensor(texts[stidx:stidx+batch_size]).float().cuda()
                features = torch.cat((u, v, torch.abs(u - v), u * v), 1)
                output = classifier(features)
                result = torch.sum(sigmoid(output)).item()
                currAuthorVotes = currAuthorVotes + result
            votes.append(currAuthorVotes)

        currPred = votes.index(max(votes))
        pred.append(currPred)
        if currPred == Ytrain[i_index]:
            correct = correct + 1

    acc =100 * correct / len(Xtrain)
    returnDict = {'cohort_train_acc': acc}
    trainPredQ.put({'cohort': pred})

    correct = 0.0
    pred = list()

    for i_index in range(len(Xtest)):
        i = Xtest[i_index]
        votes = list()

        for author in authorList:
            currAuthorVotes = 0
            texts = author[:K]
            for stidx in range(0, K, batch_size):
                u = torch.tensor(np.tile(i,(batch_size,1))).float().cuda()
                v = torch.tensor(texts[stidx:stidx+batch_size]).float().cuda()
                features = torch.cat((u, v, torch.abs(u - v), u * v), 1)
                output = classifier(features)
                result = torch.sum(sigmoid(output)).item()
                currAuthorVotes = currAuthorVotes + result
            votes.append(currAuthorVotes)

        currPred = votes.index(max(votes))
        pred.append(currPred)
        if currPred == Ytest[i_index]:
            correct = correct + 1

    acc =100 * correct / len(Xtest)
    returnDict['cohort_test_acc'] =  acc
    q.put(returnDict)
    testPredQ.put({'cohort': pred})

def nnClassifier(Xtrain, Ytrain, Xtest, Ytest, q, trainPredQ, testPredQ):
    inputdim = int(len(Xtrain[0]))
    nlabel = int(np.amax(Ytrain) + 1)
    torch.cuda.set_device(2)
    classifier = _classifier(inputdim, nlabel)

    optimizer = optim.Adam(classifier.parameters())
    criterion = nn.CrossEntropyLoss()

    classifier.cuda()
    criterion.cuda()

    train_accs = list()
    test_accs = list()
    trainPredictions = list()
    testPredictions = list()

    # training epochs
    epochs = 500
    for epoch in range(epochs):
        losses = []
        batch_size = 30
        correct = 0.0
        pred = list()
        for stidx in range(0, len(Xtrain), batch_size):
            inputv = torch.FloatTensor(Xtrain[stidx:stidx + batch_size]).cuda()
            labelsv = torch.tensor(Ytrain[stidx:stidx + batch_size]).cuda()
            k = len(Xtrain[stidx:stidx + batch_size])

            output = classifier(inputv)

            loss = criterion(output, labelsv)

            predictions = torch.max(output, 1)[1].cpu().numpy()
            pred = pred + predictions.tolist()
            for i in range(len(predictions)):
                if predictions[i] == Ytrain[stidx + i]:
                    correct = correct + 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.data.mean())
        train_accs.append(100. * correct / len(Xtrain))
        trainPredictions.append(pred)

        # test
        correct = 0.0
        pred = list()
        for stidx in range(0, len(Xtest), batch_size):
            inputv = torch.FloatTensor(Xtest[stidx:stidx + batch_size]).cuda()

            output = classifier(inputv)

            predictions = torch.max(output, 1)[1].cpu().numpy()
            pred = pred + predictions.tolist()
            for i in range(len(predictions)):
                if predictions[i] == Ytest[stidx + i]:
                    correct = correct + 1
        test_accs.append(100. * correct / len(Xtest))
        testPredictions.append(pred)


    bestEpoch = test_accs.index(max(test_accs))
    train_acc = train_accs[bestEpoch]
    test_acc = test_accs[bestEpoch]

    returnDict = {'nnet_train_acc': train_acc, 'nnet_test_acc': test_acc, 'epoch': bestEpoch}
    q.put(returnDict)
    trainPredQ.put({'nnet': trainPredictions[epoch]})
    testPredQ.put({'nnet': testPredictions[epoch]})

def readInputCSV(modelName, dataset):
    with open('encoded/'+dataset+'-encodedTrainData-'+modelName+'.csv', newline='', encoding="utf-8") as csvfile:
        data = pd.read_csv(csvfile, header=None, delimiter=',', encoding='utf-8')

        Xtrain = np.array(data.iloc[:, :-1])
        Ytrain = np.array(data.iloc[:, -1])

        authorList = list()
        currAuthor = 0
        currList = list()
        for index in range(len(Xtrain)):
            # Check if current entry is of same author
            if Ytrain[index] != currAuthor:
                # If current entry is of a different author, reset
                authorList.append(currList)
                currList = list()
                currAuthor = Ytrain[index]
            # Attribute current row to current author's list
            currList.append(Xtrain[index])
        authorList.append(currList)

    with open('encoded/'+dataset+'-encodedTestData-'+modelName+'.csv', newline='', encoding="utf-8") as csvfile:
        data = pd.read_csv(csvfile, header=None, delimiter=',', encoding='utf-8')

        Xtest = np.array(data.iloc[:, :-1])
        Ytest = np.array(data.iloc[:, -1])

        return authorList, Xtrain, Ytrain, Xtest, Ytest

class _classifier(nn.Module):
    def __init__(self, inputdim, nlabel):
        super(_classifier, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(inputdim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, nlabel),
        )

    def forward(self, input):
        return self.main(input)

def loadClassifier(modelName, dataset):
    torch.cuda.set_device(0)
    savedir = 'savedir/'
    fileName = dataset + '-' + modelName + '.pickle'
    model = torch.load(os.path.join(savedir, fileName), map_location={'cuda:2':'cuda:0'})
    classifier = model.classifier
    return classifier

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

if __name__ == "__main__":
    #dill.dump_session('classifierResults.pkl', main=main())
    main()
