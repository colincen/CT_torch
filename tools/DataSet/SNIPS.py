import os.path as op
import os
import random
import copy

class snips:
    def __init__(self, dataDir, desc_path, cross_domain=False, target_domain='PlayMusic'):
        self.data = None
        self.source = {'train': [], 'dev': [], 'test': []}
        self.target = {'train': [], 'dev': [], 'test': []}
        self.description = None
        self.dataDir = dataDir
        self.train, train_slots = self.prepareSentencesByIntent('train.txt')
        self.dev, dev_slots = self.prepareSentencesByIntent('dev.txt')
        self.test,test_slots = self.prepareSentencesByIntent('test.txt')
        self.mergeSlot()
        self.description = self.getDescription(desc_path)

        if not cross_domain:
            self.source['train'] = self.train[target_domain]
            self.source['dev'] = self.dev[target_domain]
            self.source['test'] = self.test[target_domain]
        else:
            self.target['test'] = self.test[target_domain] + self.train[target_domain] + self.dev[target_domain]
            for k, v in self.train.items():
                if k != target_domain:
                    self.source['train'] += self.train[k]
                    self.source['dev'] += self.dev[k]
                    self.source['test'] += self.test[k]
        self.data = {'source': self.source, 'target': self.target, 'description': self.description}


    def mergeSlot(self):
        slots = {}
        dd = [self.train, self.dev, self.test]
        for d in dd:
            for k,v in d.items():
                slots[k] = set()
                for i in range(len(v)):
                    for w in v[i]['slot']:
                        slots[k].add(w)
            for k in slots.keys():
                t = list(slots[k])
                slots[k] = t

            for k,v in d.items():
                for i in range(len(d[k])):
                    d[k][i]['slot'] = slots[k]



    def addNegativeSample(self, da, ratio=1):
        slots = self.description.keys()
        slot_set =set()
        slot_set.add(da[0]['slot'][0])
        neg_list = []
        for i in range(1, len(da)):
            if da[i]['tokens'] == da[i-1]['tokens']:
                slot_set.add(da[i]['slot'][0])
            else:
                temp = []
                for j in slots:
                    if j not in slot_set:
                        temp.append(j)
                slot_set.clear()
                neg_slot = random.sample(temp, ratio)
                # tempdict = {'tokens':[], 'NER_BIO':[], 'slot':[]}
                for j in range(ratio):
                    tempdict = {'tokens': [], 'NER_BIO': [], 'slot': []}
                    tempdict['tokens'] = da[i-1]['tokens']
                    tempdict['NER_BIO'] = ['O'] * len(tempdict['tokens'])
                    tempdict['slot'].append(neg_slot[j])
                    neg_list.append(tempdict)


        resdata =[]

        resdata += da
        resdata += neg_list
        return resdata

    def getRawSentences(self, path):
        rawText = []
        sentence = []
        label = []
        intent = []

        for line in open(op.join(self.dataDir, path), 'r'):
            row = line.strip().split()

            # read the blank space
            if len(row) == 0:
                rawText.append([sentence.copy(), label.copy(), intent.copy()])
                sentence.clear()
                label.clear()

            # read the intent
            elif len(row) == 1:
                intent = row

            # read the word and label
            elif len(row) == 2:
                sentence.append(row[0])
                label.append(row[1])

        rawText.append([sentence.copy(), label.copy(), intent.copy()])

        return rawText

    def prepareSentencesByIntent(self, path):
        rawData = self.getRawSentences(path)
        print('total sentence from %s is %d' % (op.join(op.basename(self.dataDir), path), len(rawData)))
        intents = set(row[2][0] for row in rawData)
        data = {intentName: [] for intentName in intents}
        slots = set()

        for sent in rawData:
            sents, slot = self.getSentenceBySlot(sent)
            data[sent[2][0]] += sents
            slots.update(slot)

        return data, slots

    def getDescription(self, path):
        slot2description = {}
        for line in open(path, 'r'):
            pos = line.find(':')
            slot = line[:pos].strip(' ')
            desc = line[pos + 1:].strip().split(' ')
            slot2description[slot] = desc
        return slot2description


    def getSentenceBySlot(self, row):
        sent, labels, intent = row[0], row[1], row[2]
        res = []
        labelSet = set()
        for label in labels:
            if label != 'O':
                labelSet.add(label[2:])
        for label in labelSet:
            tempDict = {'tokens': sent,
                        'NER_BIO': [lab[:1] if len(lab) > 1 and lab[2:] == label else 'O' for lab in labels],
                        'Raw_labels' : labels,
                        'slot': [label]}
            res.append(tempDict)

        return res, labelSet

if __name__ == '__main__':
    dataDir = '/home/sh/data/JointSLU-DataSet/formal_snips'
    desc_path = '/home/sh/code/CT_torch/data/snips_slot_description.txt'
    s = snips(dataDir=dataDir, desc_path=desc_path, cross_domain=True)
