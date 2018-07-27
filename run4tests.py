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

def main():
    dataset = input("Dataset name: ")
    modelName = input("Model name: ")
    description = input("Description: ")
    global Ytest
    authorList, Xtrain, Ytrain, Xtest, Ytest = readInputCSV(modelName, dataset)
    print("Data successfully loaded.")

    q = Queue()
    predQ = Queue()

    numAlgs = 4
    svm_process = Process(target=SVM, args=(modelName, Xtrain, Ytrain, Xtest, Ytest, q, dataset, predQ))
    knn_process = Process(target=kNN, args=(5, Xtrain, Ytrain, Xtest, Ytest, q, predQ))
    cohort_process = Process(target=cohort, args=(30, modelName, authorList, Xtest, Ytest, q, dataset, predQ))
    nnet_process = Process(target=nnClassifier, args=(Xtrain, Ytrain, Xtest, Ytest, q, predQ))

    svm_process.start()
    knn_process.start()
    cohort_process.start()
    nnet_process.start()

    for alg in range(numAlgs):
        print(q.get())

    predDict = {}
    for alg in range(numAlgs):
        currDict = predQ.get()
        predDict.update(currDict)

    predArray = np.vstack((np.array(predDict['svm']), np.array(predDict['knn']), np.array(predDict['nnet']), np.array(predDict['cohort'])))
    predArray = predArray.transpose()

    global correctDict
    correctDict = {}

    for index in range(len(predArray)):
        correctKey = ''

        for alg in range(numAlgs):
            if predArray[index][alg] == Ytest[index]:
                # correctly classified by this alg
                correctKey += str(alg)

        if correctKey not in correctDict:
            correctDict[correctKey] = list()

        correctDict[correctKey].append(index)

    print('Correct:')
    for key, value in correctDict.items():
        print(key + ': ' + str(len(value)))

    authorErrorDict = {}
    for index in correctDict['']:
        if str(Ytest[index]) not in authorErrorDict:
            authorErrorDict[str(Ytest[index])] = 1
        else:
            authorErrorDict[str(Ytest[index])] += 1


    print('authorErrorDict: ')
    print(authorErrorDict)
    print('Authors with more than 25 texts misclassified by all algorithms: ')
    for key, value in authorErrorDict.items():
        if value > 24:
            print(key + ": " + str(value))


    print("multi-class classification results for "+ color.RED + modelName + color.END + ":")
    print("Description: " + description)



def SVM(modelName, Xtrain, Ytrain, Xtest, Ytest, q, dataset, predQ):
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

    predictions = mySVM.predict(Xtest)
    correct = 0.0
    for pred, label in zip(predictions, Ytest):
        if pred == label:
            correct = correct + 1
    test_acc =100 * correct / len(Ytest)

    returnDict = {'svm_train_acc': train_acc, 'svm_test_acc': test_acc}
    q.put(returnDict)
    predQ.put({'svm': predictions})

def kNN(k, Xtrain, Ytrain, Xtest, Ytest, q, predQ):
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(Xtrain, Ytrain)

    predictions = neigh.predict(Xtrain)
    correct = 0.0
    for pred, label in zip(predictions, Ytrain):
        if pred == label:
            correct = correct + 1
    train_acc = 100 * correct / len(Ytrain)

    predictions = neigh.predict(Xtest)
    correct = 0.0
    for pred, label in zip(predictions, Ytest):
        if pred == label:
            correct = correct + 1
    test_acc = 100 * correct / len(Ytest)

    returnDict = {'knn_train_acc': train_acc, 'knn_test_acc': test_acc}
    q.put(returnDict)
    predQ.put({'knn': predictions})

def cohort(K, modelName, authorList, Xtest, Ytest, q, dataset, predQ):
    classifier = loadClassifier(modelName, dataset)
    print("Classifier successfully loaded.")
    correct = 0.0
    sigmoid = nn.Sigmoid()
    pred = list()
    batch_size = 30

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
    returnDict = {'cohort_test_acc': acc}
    q.put(returnDict)
    predQ.put({'cohort': pred})

def nnClassifier(Xtrain, Ytrain, Xtest, Ytest, q, predQ):
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

    # training epochs
    epochs = 500
    for epoch in range(epochs):
        losses = []
        batch_size = 30
        correct = 0

        for stidx in range(0, len(Xtrain), batch_size):
            inputv = torch.FloatTensor(Xtrain[stidx:stidx + batch_size]).cuda()
            labelsv = torch.tensor(Ytrain[stidx:stidx + batch_size]).cuda()
            k = len(Xtrain[stidx:stidx + batch_size])

            output = classifier(inputv)

            loss = criterion(output, labelsv)

            predictions = torch.max(output, 1)[1].cpu().numpy()
            for i in range(len(predictions)):
                if predictions[i] == Ytrain[stidx + i]:
                    correct = correct + 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.data.mean())
        train_accs.append(100. * correct / len(Xtrain))

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


    bestEpoch = test_accs.index(max(test_accs))
    train_acc = train_accs[bestEpoch]
    test_acc = test_accs[bestEpoch]

    returnDict = {'nnet_train_acc': train_acc, 'nnet_test_acc': test_acc, 'epoch': bestEpoch}
    q.put(returnDict)
    predQ.put({'nnet': pred})

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
    main()