import numpy as np
import pandas as pd
from sklearn import svm, tree, naive_bayes, linear_model
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelBinarizer
from multiprocessing import Process, Queue
import models
import torch.nn as nn
import torch
import torch.optim as optim
import os
import pdb
import dill
import xgboost as xgb


def main():
    dataset = input("Dataset name: ")
    modelName = input("Model name: ")
    global Ytest
    authorList, Xtrain, Ytrain, Xtest, Ytest = readInputCSV(modelName, dataset)
    print("Data successfully loaded.")

    q = Queue()
    trainPredQ = Queue()
    testPredQ = Queue()

    numAlgs = 2
    svm_process = Process(target=SVM, args=(modelName, Xtrain, Ytrain, Xtest, Ytest, q, dataset, trainPredQ, testPredQ))
    knn_process = Process(target=kNN, args=(5, Xtrain, Ytrain, Xtest, Ytest, q, trainPredQ, testPredQ))
    #cohort_process = Process(target=cohort, args=(30, modelName, authorList, Xtrain, Ytrain, Xtest, Ytest, q, dataset, trainPredQ, testPredQ))
    #nnet_process = Process(target=nnClassifier, args=(Xtrain, Ytrain, Xtest, Ytest, q, trainPredQ, testPredQ))

    svm_process.start()
    knn_process.start()
    #cohort_process.start()
    #nnet_process.start()

    # Save accuracy of each alg in a dict
    accDict = {}
    for alg in range(numAlgs):
        currDict = q.get()
        accDict.update(currDict)
    accuracies = [accDict['svm_train_acc'], accDict['knn_train_acc']]#, accDict['nnet_train_acc'], accDict['cohort_train_acc']]

    # Put predictions into dictionaries
    trainPreds = {}
    testPreds = {}
    for alg in range(numAlgs):
        currTrainDict = trainPredQ.get()
        trainPreds.update(currTrainDict)
        currTestDict = testPredQ.get()
        testPreds.update(currTestDict)

    # Put predictions into big matrices
    trainPredArray = np.vstack((np.array(trainPreds['svm']), np.array(trainPreds['knn'])))#, np.array(trainPreds['nnet']), np.array(trainPreds['cohort'])))
    trainPredArray = trainPredArray.transpose()


    testPredArray = np.vstack((np.array(testPreds['svm']), np.array(testPreds['knn'])))#, np.array(testPreds['nnet']), np.array(testPreds['cohort'])))
    testPredArray = testPredArray.transpose()

    # List of algs that correctly classify each algorithm
    correctlyClassified = list()
    classDict = {}
    for index in range(len(Xtrain)):
        correctAlgs = ''
        for alg in range(numAlgs):
            if trainPredArray[index][alg] == Ytrain[index]:
                # correctly classified by this alg
                correctAlgs += str(alg)
        # Use a dictionary to convert Str classes into unique numbers
        if correctAlgs not in classDict:
            classDict[correctAlgs] = len(classDict)
        correctlyClassified.append(classDict[correctAlgs])





    #dill.load_session('classifierResults.pkl')
    # labelled data to train a tree (Xtrain[index], correctlyClassified[index])
    n_classes = len(classDict)
    '''
    algClassifier = tree.DecisionTreeClassifier(max_depth=265)
    algClassifier = algClassifier.fit(Xtrain, correctlyClassified)
    '''
    algClassifier = xgb.XGBClassifier(max_depth=256)
    algClassifier = algClassifier.fit(Xtrain, correctlyClassified)
    print('Tree fit')

    # predict classifier for Xtest
    classifierPredictions = algClassifier.predict(Xtest)

    # print alg classifier train acc
    correct = 0.0
    for index in range(len(classifierPredictions)):
        if classifierPredictions[index] == correctlyClassified[index]:
            correct += 1
    print(str(100*correct/len(correctlyClassified)))
    # Use class dict to convert alg predictions back into strings so we can use them
    reverseClassDict = {v: k for k, v in classDict.items()}

    strClassifierPredictions = [reverseClassDict[i] for i in classifierPredictions]


    assert(len(strClassifierPredictions) == len(Xtest))

    outputPredictions = []
    for index in range(len(Xtest)):

        if len(strClassifierPredictions[index]) == 0:
            # No classifier is good for this text
            # Use KNN
            outputPredictions.append(testPredArray[index][1])
        elif len(strClassifierPredictions[index]) == 1:
            # Only one classifier is good for this type of data so use that one
            alg = int(strClassifierPredictions[index])
            outputPredictions.append(testPredArray[index][alg])
        else:
            # Alg predicts that multiple classifiers could predict this correctly
            # See if they all agree, there is a majority vote, or they all disagree
            currPredVotes = {}
            for alg in strClassifierPredictions[index]:
                if str(testPredArray[index][int(alg)]) not in currPredVotes:
                    currPredVotes[str(testPredArray[index][int(alg)])] = 1
                else:
                    currPredVotes[str(testPredArray[index][int(alg)])] += 1

            # find index w most votes
            maxValue = max(currPredVotes.values())  # maximum value
            maxKeys = [k for k, v in currPredVotes.items() if v == maxValue]

            if len(maxKeys) > 1:
                # There is a tie, choose knn
                outputPredictions.append(testPredArray[index][1])
                
                # There is a tie, choose one with highest overall acc
                #currHighestAcc = 0
                #currVote = -1
                #for alg in strClassifierPredictions[index]:
                #    if accuracies[int(alg)] > currHighestAcc:
                #        currHighestAcc = accuracies[int(alg)]
                #        currVote = int(alg)

                #outputPredictions.append(testPredArray[index][currVote])
                
            else:
                # Choose the one with the most votes
                outputPredictions.append(int(maxKeys[0]))

    assert (len(outputPredictions) == len(Xtest))
    # Calculate accuracy of ensemble method
    correct = 0.0
    for index in range(len(Ytest)):
        if outputPredictions[index] == Ytest[index]:
            correct += 1
    acc = 100 * correct/len(Ytest)
    print("Ensemble acc: " + str(acc))
    print("multi-class classification results for "+ color.RED + modelName + color.END + ":")



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
