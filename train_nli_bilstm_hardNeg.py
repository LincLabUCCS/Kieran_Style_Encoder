# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import sys
import time
import argparse

import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn

from data_contd import get_nli, get_batch, build_vocab, splitSent
from mutils import get_optimizer
from models import NLINet

import random




parser = argparse.ArgumentParser(description='NLI training')

parser.add_argument("--outputmodelname", type=str, default='biLSTM256-3grams64clsfr-lemb2-hardNeg2.pickle')
parser.add_argument("--dataset_name", type=str, default='C50')
parser.add_argument("--n_epochs", type=int, default=40)
parser.add_argument("--batch_size", type=int, default=12)
parser.add_argument("--dpout_fc", type=float, default=0.2, help="classifier dropout")
parser.add_argument("--encoder_type", type=str, default='BLSTMEncoder', help="see list of encoders")
parser.add_argument("--enc_lstm_dim", type=int, default=128, help="encoder nhid dimension")
parser.add_argument("--n_enc_layers", type=int, default=1, help="encoder num layers")
parser.add_argument("--fc_dim", type=int, default=64, help="nhid of fc layers")
parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
parser.add_argument("--split_type", type=str, default='n-grams', help="GPU ID")
parser.add_argument("--embedding_type", type=str, default='learned')
parser.add_argument("--pool_type", type=str, default='max', help="max or mean")


# paths
parser.add_argument("--nlipath", type=str, default='dataset/', help="NLI data path (SNLI or MultiNLI)")
parser.add_argument("--outputdir", type=str, default='savedir/', help="Output directory")



# training
parser.add_argument("--dpout_model", type=float, default=0., help="encoder dropout")

parser.add_argument("--nonlinear_fc", type=float, default=1, help="use nonlinearity in fc")
parser.add_argument("--optimizer", type=str, default="adam", help="adam or sgd,lr=0.1")
parser.add_argument("--lrshrink", type=float, default=5, help="shrink factor for sgd")
parser.add_argument("--decay", type=float, default=0.99, help="lr decay")
parser.add_argument("--minlr", type=float, default=1e-5, help="minimum lr")
parser.add_argument("--max_norm", type=float, default=5., help="max norm (grad clipping)")

# model


parser.add_argument("--n_classes", type=int, default=1, help="entailment/neutral/contradiction")

# gpu

parser.add_argument("--seed", type=int, default=1234, help="seed")



params, _ = parser.parse_known_args()
params.outputmodelname = params.dataset_name + '-' + params.outputmodelname

#glove path
if params.split_type == 'n-grams':
    if params.embedding_type == 'learned':
        GLOVE_PATH = "dataset/GloVe/C50-traintest-3grams.txt"
    else:
        GLOVE_PATH = "dataset/GloVe/charNgram.txt"

    params.word_emb_dim = 100
else:
    if params.embedding_type == 'learned':
        GLOVE_PATH = "dataset/GloVe/traintestwordmodel.txt"
        params.word_emb_dim = 100
    else:
        GLOVE_PATH = "dataset/GloVe/glove.840B.300d.txt"
        params.word_emb_dim = 300


# set gpu device
torch.cuda.set_device(params.gpu_id)

# print parameters passed, and all parameters
print('\ntogrep : {0}\n'.format(sys.argv[1:]))
print(params)


"""
SEED
"""
np.random.seed(params.seed)
torch.manual_seed(params.seed)
torch.cuda.manual_seed(params.seed)

"""
DATA
"""
train, valid, test = get_nli(params.nlipath, params.dataset_name)
word_vec = build_vocab(train['texts'] + valid['s1'] + valid['s2'] +
                       test['s1'] + test['s2'], GLOVE_PATH, splitType=params.split_type)

authorList = list()

for author in train['authorList']:
    authorList.append(np.array([[word for word in splitSent(text, splitType=params.split_type) if word in word_vec] for text in author]))

for split in ['s1', 's2']:
    for data_type in ['valid', 'test']:
        eval(data_type)[split] = np.array([[word for word in splitSent(sent, splitType=params.split_type) if word in word_vec] for sent in eval(data_type)[split]])




"""
MODEL
"""
# model config
config_nli_model = {
    'n_words'        :  len(word_vec)          ,
    'word_emb_dim'   :  params.word_emb_dim   ,
    'enc_lstm_dim'   :  params.enc_lstm_dim   ,
    'n_enc_layers'   :  params.n_enc_layers   ,
    'dpout_model'    :  params.dpout_model    ,
    'dpout_fc'       :  params.dpout_fc       ,
    'fc_dim'         :  params.fc_dim         ,
    'bsize'          :  params.batch_size     ,
    'n_classes'      :  params.n_classes      ,
    'pool_type'      :  params.pool_type      ,
    'nonlinear_fc'   :  params.nonlinear_fc   ,
    'encoder_type'   :  params.encoder_type   ,
    'use_cuda'       :  True                  ,
    'split_type'     :  params.split_type     ,

}

# model
encoder_types = ['BLSTMEncoder', 'BLSTMprojEncoder', 'BGRUlastEncoder',
                 'InnerAttentionMILAEncoder', 'InnerAttentionYANGEncoder',
                 'InnerAttentionNAACLEncoder', 'ConvNetEncoder', 'LSTMEncoder']
assert params.encoder_type in encoder_types, "encoder_type must be in " + \
                                             str(encoder_types)
# Start from scratch
nli_net = NLINet(config_nli_model)
# Load
#nli_net = torch.load(os.path.join(params.outputdir, params.outputmodelname))
print(nli_net)

# loss
weight = torch.FloatTensor(params.n_classes).fill_(1)
loss_fn = nn.BCEWithLogitsLoss(weight=weight)
loss_fn.size_average = False

# optimizer
optim_fn, optim_params = get_optimizer(params.optimizer)
optimizer = optim_fn(nli_net.parameters(), **optim_params)

# cuda by default
nli_net.cuda()
loss_fn.cuda()


"""
TRAIN
"""
val_f1_best = -1e10
adam_stop = False
stop_training = False
lr = optim_params['lr'] if 'sgd' in params.optimizer else None


def trainepoch(epoch):
    print('\nTRAINING : Epoch ' + str(epoch))
    nli_net.train()
    all_costs = []
    logs = []
    words_count = 0

    last_time = time.time()
    correct = 0.
    truePos = 0.
    falsePos = 0.
    trueNeg = 0.
    falseNeg = 0.
    precision = 0
    recall = 0

    # shuffle the data

    # Produce pairs from train data

    s1 = list()
    s2 = list()
    text1Author = list()
    text2Author = list()
    target = list()

    numAuthors = len(authorList)

    if epoch == 1:
        DiffPairsPerAuthor = 100
        SamePairsPerAuthor = DiffPairsPerAuthor
    else:
        DiffPairsPerAuthor = 10
        SamePairsPerAuthor = 50

    for authorIndex in range(numAuthors):
        for n in range(DiffPairsPerAuthor):
            # Choose text from current author's works
            textIndex = random.randint(0, len(authorList[authorIndex])-1)

            # Choose pair author
            pairAuthorIndex = authorIndex
            while pairAuthorIndex == authorIndex:
                pairAuthorIndex = random.randint(0, numAuthors-1)

            # Choose text from pair author's works
            pairTextIndex = random.randint(0, len(authorList[pairAuthorIndex])-1)

            # Output pair
            s1.append(authorList[authorIndex][textIndex])
            s2.append(authorList[pairAuthorIndex][pairTextIndex])
            text1Author.append(authorIndex)
            text2Author.append(pairAuthorIndex)
            target.append([0])

        for n in range(SamePairsPerAuthor):
            # Choose text from current author's works
            textIndex = random.randint(0, len(authorList[authorIndex])-1)

            # Choose pair author
            pairAuthorIndex = authorIndex

            # Choose text from pair author's works
            pairTextIndex = random.randint(0, len(authorList[pairAuthorIndex])-1)

            # Output pair
            s1.append(authorList[authorIndex][textIndex])
            s2.append(authorList[pairAuthorIndex][pairTextIndex])
            text1Author.append(authorIndex)
            text2Author.append(pairAuthorIndex)
            target.append([1])


    for authorPair in hardPairs[epoch-1]:
        numExamples = 10 * hardPairs[epoch - 1][authorPair]
        if numExamples > 100:
            numExamples = 100
        author1 = int(authorPair.split('-')[0])
        author2 = int(authorPair.split('-')[1])
        for n in range(numExamples):
            textIndex = random.randint(0, len(authorList[author1]) - 1)
            pairTextIndex = random.randint(0, len(authorList[author2]) - 1)
            # Output pair
            s1.append(authorList[author1][textIndex])
            s2.append(authorList[author2][pairTextIndex])
            text1Author.append(author1)
            text2Author.append(author2)
            if author1 == author2:
                target.append([1])
            else:
                target.append([0])




    #for index in range(len(text2Author)):
    #    print(str(text1Author[index]) + ' && ' + str(text2Author[index]))

    # shuffle
    permutation = np.random.permutation(len(s1))
    s1 = np.array(s1)[permutation]
    s2 = np.array(s2)[permutation]
    text1Author = np.array(text1Author)[permutation]
    text2Author = np.array(text2Author)[permutation]
    target = np.array(target)[permutation]



    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * params.decay if epoch>1\
        and 'sgd' in params.optimizer else optimizer.param_groups[0]['lr']
    print('Learning rate : {0}'.format(optimizer.param_groups[0]['lr']))
    print('Training Pairs: ' + str(len(s1)))
    correct = 0


    for stidx in range(0, len(s1), params.batch_size):
        # prepare batch
        s1_batch, s1_len = get_batch(s1[stidx:stidx + params.batch_size],
                                     word_vec, params.word_emb_dim)
        s2_batch, s2_len = get_batch(s2[stidx:stidx + params.batch_size],
                                     word_vec, params.word_emb_dim)
        s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())
        tgt_batch = torch.FloatTensor(target[stidx:stidx + params.batch_size]).cuda()
        # Uncomment next 2 lines to go from hardNeg2 to hardNeg
        #loss_weights =  torch.ones_like(tgt_batch).cuda() + tgt_batch
        #loss_fn.weight = loss_weights

        k = s1_batch.size(1)  # actual batch size

        # model forward
        output = nli_net((s1_batch, s1_len), (s2_batch, s2_len))

        for i in range(len(output)):
            if output[i].item() < 0:
                pred = 0
            else:
                pred = 1

            if text1Author[stidx + i] < text2Author[stidx + i]:
                authorPairString = str(text1Author[stidx + i]) + '-' + str(text2Author[stidx + i])
            else:
                authorPairString = str(text2Author[stidx + i]) + '-' + str(text1Author[stidx + i])

            if pred == 1:
                # pred pos
                if pred == int(tgt_batch[i].item()):
                    truePos = truePos + 1
                    correct = correct + 1
                else:
                    falsePos = falsePos + 1
                    # incorrectly classified
                    if authorPairString in hardPairs[epoch]:
                        hardPairs[epoch][authorPairString] = hardPairs[epoch][authorPairString] + 1
                    else:
                        hardPairs[epoch][authorPairString] = 1
            else:
                #pred neg
                if pred == int(tgt_batch[i].item()):
                    trueNeg = trueNeg + 1
                    correct = correct + 1
                else:
                    falseNeg = falseNeg + 1
                    # incorrectly classified
                    if authorPairString in hardPairs[epoch]:
                        hardPairs[epoch][authorPairString] = hardPairs[epoch][authorPairString] + 1
                    else:
                        hardPairs[epoch][authorPairString] = 1

        # loss
        loss = loss_fn(output, tgt_batch)
        all_costs.append(loss.data.item())
        words_count += (s1_batch.nelement() + s2_batch.nelement()) / params.word_emb_dim

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient clipping (off by default)
        shrink_factor = 1
        total_norm = 0

        for p in nli_net.parameters():
            if p.requires_grad:
                p.grad.data.div_(k)  # divide by the actual batch size
                total_norm += p.grad.data.norm() ** 2
        total_norm = np.sqrt(total_norm)

        if total_norm > params.max_norm:
            shrink_factor = params.max_norm / total_norm
        current_lr = optimizer.param_groups[0]['lr'] # current lr (no external "lr", for adam)
        optimizer.param_groups[0]['lr'] = current_lr * shrink_factor # just for update

        # optimizer step
        optimizer.step()
        optimizer.param_groups[0]['lr'] = current_lr

        if (truePos + falsePos) > 0:
            precision = 100*truePos / (truePos + falsePos)
        else:
            precision = 0
        if (truePos + falseNeg) > 0:
            recall = 100*truePos / (truePos + falseNeg)
        else:
            recall = 0
        if not precision == 0 and not recall == 0:
            f1 = 2*(precision * recall)/(precision + recall)
        else:
            f1 = 0

        if len(all_costs) == 20:
            logs.append('{0} ; loss {1} ; sentence/s {2} ; words/s {3} ; accuracy train : {4} ; precision : {5} ; recall {6} ; F1: {7}'.format(
                            stidx, round(np.mean(all_costs), 2),
                            int(len(all_costs) * params.batch_size / (time.time() - last_time)),
                            int(words_count * 1.0 / (time.time() - last_time)),
                            100.*correct/(stidx+k), precision, recall, f1))
            print(logs[-1])
            last_time = time.time()
            words_count = 0
            all_costs = []

    train_acc = 100 * correct/len(s1)
    print('results : epoch {0} ; mean accuracy train : {1}'
          .format(epoch, train_acc))
    print(hardPairs[epoch])
    print(len(hardPairs[epoch]))
    return train_acc


def evaluate(epoch, eval_type='valid', final_eval=False):
    nli_net.eval()
    correct = 0.
    truePos = 0.
    falsePos = 0.
    trueNeg = 0.
    falseNeg = 0.
    global val_f1_best, lr, stop_training, adam_stop

    if eval_type == 'valid':
        print('\nVALIDATION : Epoch {0}'.format(epoch))

    s1 = valid['s1'] if eval_type == 'valid' else test['s1']
    s2 = valid['s2'] if eval_type == 'valid' else test['s2']
    target = valid['label'] if eval_type == 'valid' else test['label']

    for i in range(0, len(s1), params.batch_size):
        # prepare batch
        s1_batch, s1_len = get_batch(s1[i:i + params.batch_size], word_vec, params.word_emb_dim)
        s2_batch, s2_len = get_batch(s2[i:i + params.batch_size], word_vec, params.word_emb_dim)
        s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())
        tgt_batch = Variable(torch.LongTensor(target[i:i + params.batch_size])).cuda()

        # model forward
        output = nli_net((s1_batch, s1_len), (s2_batch, s2_len))

        for i in range(len(output)):
            if output[i].item() < 0:
                pred = 0
            else:
                pred = 1
            
            if pred == 1:
                # pred pos
                if pred == int(tgt_batch[i].item()):
                    truePos = truePos + 1
                    correct = correct + 1
                else:
                    falsePos = falsePos + 1
            else:
                #pred neg
                if pred == int(tgt_batch[i].item()):
                    trueNeg = trueNeg + 1
                    correct = correct + 1
                else:
                    falseNeg = falseNeg + 1



    # save model
    eval_acc = 100 * correct / len(s1)
    if (truePos + falsePos) > 0:
        precision = 100 * truePos / (truePos + falsePos)
    else:
        precision = 0
    if (truePos + falseNeg) > 0:
        recall = 100 * truePos / (truePos + falseNeg)
    else:
        recall = 0
    if not precision == 0 and not recall == 0:
        f1 =  2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0

    if final_eval:
        print('finalgrep : accuracy {0} : {1} ; precision: {2} ; recall: {3} ; F1: {4}'.format(eval_type, eval_acc, precision, recall, f1))
    else:
        print('togrep : results : epoch {0} ; mean accuracy {1} : {2} ; precision: {3} ; recall: {4} ; F1: {5}'.format(epoch, eval_type, eval_acc, precision, recall, f1))

    if eval_type == 'valid' and epoch <= params.n_epochs:
        if f1 >= val_f1_best:
            print('saving model at epoch {0}'.format(epoch))
            if not os.path.exists(params.outputdir):
                os.makedirs(params.outputdir)
            torch.save(nli_net, os.path.join(params.outputdir,
                       params.outputmodelname))
            val_f1_best = f1
        else:
            if 'sgd' in params.optimizer:
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / params.lrshrink
                print('Shrinking lr by : {0}. New lr = {1}'
                      .format(params.lrshrink,
                              optimizer.param_groups[0]['lr']))
                if optimizer.param_groups[0]['lr'] < params.minlr:
                    stop_training = True
            if 'adam' in params.optimizer:
                # early stopping (at 2nd decrease in accuracy)
                stop_training = adam_stop
                #adam_stop = True
    return eval_acc


"""
Train model on Natural Language Inference task
"""
epoch = 1
global hardPairs
hardPairs = list()
hardPairs.append({})


while not stop_training and epoch <= params.n_epochs:
    hardPairs.append({})
    train_acc = trainepoch(epoch)
    eval_acc = evaluate(epoch, 'valid')
    epoch += 1

# Run best model on test set.
del nli_net
nli_net = torch.load(os.path.join(params.outputdir, params.outputmodelname))

print('\nTEST : Epoch {0}'.format(epoch))
evaluate(1e6, 'valid', True)
evaluate(0, 'test', True)

# Save encoder instead of full model
torch.save(nli_net.encoder,
           os.path.join(params.outputdir, params.outputmodelname + '.encoder'))
