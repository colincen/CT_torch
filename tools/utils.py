import os
import os.path as op
import gzip
from typing import List

import numpy as np
import json
import random
import copy
import math
import tools.DataSet.SNIPS as SNIPS

def cal_maxlen(data):
    return max([len(x) for x in data])

def padData(data, max_len, padding_idx):
    padded = []
    for i in range(len(data)):
        temp = []
        pad = [padding_idx] * (max_len - len(data[i]))
        for token in data[i]:
            temp.append(token)
        temp += pad
        padded.append(temp)
    return padded


def prepare_data(sents):
    sentences = []
    for i in sents:
        if len(i) > 0:
            sentences.append(i)
    sentences = sorted(sentences, key=lambda x: len(x), reverse=True)
    lengths = [len(i) for i in sentences]
    return sentences, lengths


def getNERdata(dataSetName='SNIPS',
               dataDir='/home/sh/data/JointSLU-DataSet/formal_snips',
               desc_path='../data/snips_slot_description.txt',
               cross_domain=True,
               target_domain='PlayMusic'):
    if not os.path.exists(dataDir):
        raise Exception('data file not exists')

    if dataSetName == 'SNIPS':
        snips = SNIPS.snips(dataDir=dataDir, desc_path=desc_path, cross_domain=True,
                                          target_domain=target_domain)
        return snips.data


def ExtractLabelsFromTokens(data):
    Labels = {}
    for row in data:
        for token in row[1]:
            if token not in Labels:
                Labels[token] = len(Labels)

    return Labels


def readTokenEmbeddings(embeddingsPath):
    if not op.isfile(embeddingsPath):
        print('Embedding not found : Error')
    word2Idx = {}
    embeddings = []
    neededVocab = {}

    embeddingsIn = gzip.open(embeddingsPath, "rt") if embeddingsPath.endswith('.gz') else open(embeddingsPath,
                                                                                               encoding="utf8")
    embeddingsDimension = None
    for line in embeddingsIn:
        split = line.rstrip().split(" ")
        word = split[0]

        if embeddingsDimension == None:
            embeddingsDimension = len(split) - 1

        if (len(
                split) - 1) != embeddingsDimension:  # Assure that all lines in the embeddings file are of the same length
            print("ERROR: A line in the embeddings file had more or less  dimensions than expected. Skip token.")
            continue

        if len(word2Idx) == 0:  # Add padding+unknown
            word2Idx["<PAD>"] = len(word2Idx)
            vector = np.zeros(embeddingsDimension)
            embeddings.append(vector)

            word2Idx["<UNK>"] = len(word2Idx)
            vector = np.random.uniform(-0.25, 0.25, embeddingsDimension)  # Alternativ -sqrt(3/dim) ... sqrt(3/dim)
            embeddings.append(vector)

        vector = np.array([float(num) for num in split[1:]])

        if len(neededVocab) == 0 or word in neededVocab:
            if word not in word2Idx:
                embeddings.append(vector)
                word2Idx[word] = len(word2Idx)
    embeddings = np.array(embeddings)

    return embeddings, word2Idx



def setMapping(data, mapping):
    resData = []
    for line in data:
        temp = []
        for token in line:
            if token in mapping:
                temp.append(mapping[token])
            else:
                temp.append(mapping['<UNK>'])
        resData.append(temp)
    return resData


def data_generator(data, batch_size):
    index = [i for i in range(len(data))]
    random.shuffle(index)
    x = []
    y = []
    slots = []
    bins = math.ceil(len(data) / batch_size)
    for i in range(bins):
        x.clear()
        y.clear()
        slots.clear()
        x_ids = index[i * batch_size:min(len(index), (i + 1) * batch_size)]
        for j in x_ids:
            x.append(data[j]['tokens'])
            y.append(data[j]['NER_BIO'])
            slots.append(data[j]['slot'])
        yield x, y, slots


def test_generator(data, batch_size):
    index = [i for i in range(len(data))]
    random.shuffle(index)
    x = []
    y = []
    slots = []
    bins = math.ceil(len(data) / batch_size)
    for i in range(bins):
        x.clear()
        y.clear()
        slots.clear()
        x_ids = index[i * batch_size:min(len(index), (i + 1) * batch_size)]
        for j in x_ids:
            x.append(data[j][0])
            y.append(data[j][1])
            slots.append(data[j][2])
        yield x, y, slots