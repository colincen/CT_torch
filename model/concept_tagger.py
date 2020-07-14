import torch
import torch.nn as nn
import numpy as np
from tools.utils import setMapping, padData, cal_maxlen
import copy
import torch.nn.utils.rnn as rnn_utils
import os
import model.crf as crf

class ConceptTagger(nn.Module):
    def __init__(self, config, embedding, word2Idx, label2Idx, description):
        super(ConceptTagger, self).__init__()
        self.embed_size = config.embed_size
        self.emb = embedding
        self.word2Idx = word2Idx
        self.label2Idx = label2Idx
        self.description = description
        self.use_crf = config.crf
        self.device = config.device
        self.config = config
        self.hidden_size1 = config.hidden_size1
        self.hidden_size2 = config.hidden_size2


        self.TokenEmbedding = nn.Embedding.from_pretrained(torch.from_numpy(embedding.astype(np.float32)), padding_idx=word2Idx['<PAD>'])

        self.LabelEmbedding = ConceptTagger.BuildLabelEmbedding(embedding, word2Idx, label2Idx, description,

                                                                        config.embedding_method, config.encoder_method,self.device)

        self.lstm1 = nn.LSTM(self.embed_size, self.hidden_size1, batch_first=True, bias=True, bidirectional=True)
        self.lstm2 = nn.LSTM(self.hidden_size1*2+self.embed_size, self.hidden_size2, batch_first=True, bias=True, bidirectional=True)
        self.h_projection = nn.Linear(2 * self.hidden_size2,
                                      self.LabelEmbedding.size(1),
                                      bias=True)
        self.dropout = nn.Dropout(config.dropout)
        if self.use_crf:
            self.crf = crf.CRF(labelEmbedding=self.LabelEmbedding, num_tags=self.LabelEmbedding.size(0),
                               batch_first=True)

    @staticmethod
    def BuildLabelEmbedding(embedding, word2Idx, label2Idx, description,  embedding_method,
                            encoder_method, device):
        _I = np.array([1, 0, 0]).astype(np.float32)
        _O = np.array([0, 1, 0]).astype(np.float32)
        _B = np.array([0, 0, 1]).astype(np.float32)
        embeddingDim = np.shape(embedding)[1]
        Idx2label = {v: k for k, v in label2Idx.items()}
        src_labels = [Idx2label[i] for i in range(len(Idx2label))]
        labelembedding = []
        if embedding_method == 'description':
            slot2Id = {}
            for label in src_labels:
                if len(label) > 2 and label[1] == '-':
                    slot = label[2:]
                    descs = description[slot]
                    if slot not in slot2Id:
                        slot2Id[slot] = []
                        for token in descs:
                            slot2Id[slot].append(word2Idx[token])

            if encoder_method == 'wordembedding':
                for label in src_labels:
                    if label == 'O':
                        labelembedding.append(np.concatenate((_O, np.zeros(embeddingDim)), 0))
                    else:
                        v0 = None
                        if label[0] == 'B':
                            v0 = _B
                        elif label[0] == 'I':
                            v0 = _I
                        else:
                            v0 = _O
                        temp = []
                        for t in slot2Id[label[2:]]:
                            temp.append(embedding[t])
                        temp = sum(temp) / len(temp)
                        labelembedding.append(np.concatenate((v0, temp), 0))

        labelembedding = torch.tensor(labelembedding, dtype=torch.float32, device=device)
        labelembedding.requires_grad = False
        return labelembedding



    def Eval(self, x, slot):
        _x = copy.deepcopy(x)
        _slot = copy.deepcopy(slot)
        x = setMapping(x, self.word2Idx)
        lengths = [len(i) for i in x]
        slot = [self.description[i] for i in slot]

        slot = setMapping(slot, self.word2Idx)

        slot_len = [len(i) for i in slot]

        x = padData(x, cal_maxlen(x), self.word2Idx['<PAD>'])
        slot = padData(slot, cal_maxlen(slot), self.word2Idx['<PAD>'])
        x = torch.tensor(x, device=self.device)
        mask = (x != self.word2Idx['<PAD>']).byte()
        mask.to(self.device)
        slot = torch.tensor(slot, device=self.device)
        slot_len = torch.tensor(slot_len, device=self.device)
        x = self.embedding(x)
        slot = self.embedding(slot)
        packed = rnn_utils.pack_padded_sequence(input=x, lengths=lengths, batch_first=True, enforce_sorted=False)
        enc_hiddens, (last_hidden, last_cell) = self.lstm1(packed)
        x = rnn_utils.pad_packed_sequence(enc_hiddens, batch_first=True)[0]
        # print(slot.size())
        slot = torch.sum(slot, 1)
        # print(slot.size())
        slot_embedding = slot / slot_len.unsqueeze(1).type_as(slot)
        slot_embedding = slot_embedding.unsqueeze(1)
        slot_embedding = slot_embedding.expand(-1, x.size(1), -1)
        x = torch.cat((x, slot_embedding), -1)
        packed = rnn_utils.pack_padded_sequence(input=x, lengths=lengths, batch_first=True,
                                                enforce_sorted=False)
        enc_hiddens, (last_hidden, last_cell) = self.lstm2(packed)
        x = rnn_utils.pad_packed_sequence(enc_hiddens, batch_first=True)[0]
        x = self.dropout(x)
        x = self.fc(x)

        y_pad = None
        if not self.use_crf:
            y_pad = x.argmax(-1).detach().tolist()
        else:
            y_pad = self.crf.decode(x, mask)
        pred = []
        id2label = {v: k for k, v in self.label2Idx.items()}
        for i in range(len(y_pad)):
            for j in range(len(y_pad[i])):
                y_pad[i][j] = id2label[y_pad[i][j]]
            pred.append(y_pad[i][:lengths[i]])
        for i in range(len(pred)):
            for j in range(len(pred[i])):
                if pred[i][j] != 'O':
                    pred[i][j] = pred[i][j] + '-' + _slot[i]

        return pred

    def forward(self, x, y, slot, Type):
        _x = copy.deepcopy(x)
        _y = copy.deepcopy(y)
        _slot = copy.deepcopy(slot)




        x = setMapping(x, self.word2Idx)
        lengths = [len(i) for i in x]
        y = setMapping(y, self.label2Idx)
        slot = [self.description[i[0]] for i in slot]

        slot = setMapping(slot, self.word2Idx)

        slot_len = [len(i) for i in slot]

        x = padData(x,  self.word2Idx['<PAD>'])
        y = padData(y,  self.label2Idx['O'])
        slot = padData(slot, self.word2Idx['<PAD>'])
        x = torch.tensor(x, device=self.device)
        y = torch.tensor(y, device=self.device)
        mask = (x != self.word2Idx['<PAD>']).byte()
        mask.to(self.device)
        slot = torch.tensor(slot, device=self.device)
        slot_len = torch.tensor(slot_len, device=self.device)
        x = self.TokenEmbedding(x)
        slot = self.TokenEmbedding(slot)
        packed = rnn_utils.pack_padded_sequence(input=x, lengths=lengths, batch_first=True, enforce_sorted=False)
        enc_hiddens, (last_hidden, last_cell) = self.lstm1(packed)
        x = rnn_utils.pad_packed_sequence(enc_hiddens, batch_first=True)[0]

        slot = torch.sum(slot, 1)

        slot_embedding = slot / slot_len.unsqueeze(1).type_as(slot)
        slot_embedding = slot_embedding.unsqueeze(1)
        slot_embedding = slot_embedding.expand(-1, x.size(1), -1)
        x = torch.cat((x, slot_embedding), -1)
        packed = rnn_utils.pack_padded_sequence(input=x, lengths=lengths, batch_first=True,
                                                enforce_sorted=False)
        enc_hiddens, (last_hidden, last_cell) = self.lstm2(packed)
        x = rnn_utils.pad_packed_sequence(enc_hiddens, batch_first=True)[0]
        x = self.dropout(x)
        x = self.h_projection(x)
        y_hat = torch.matmul(x, self.LabelEmbedding.transpose(0, 1))

        if Type == 'train':
            if not self.use_crf:
                batch_size, max_len = y_hat.size(0), y_hat.size(1)
                feats = y_hat.view(batch_size * max_len, -1)
                tags = y.view(-1)
                loss_func = nn.CrossEntropyLoss(size_average=True)
                loss = loss_func(feats, tags)
                return loss
            else:

                loss = -self.crf(y_hat, y, mask, 'mean')
                return loss
        elif Type == 'test':
            y_pad = None
            if not self.use_crf:
                y_pad = y_hat.argmax(-1).detach().tolist()
            else:
                y_pad = self.crf.decode(y_hat, mask)
            pred = []
            id2label = {v: k for k, v in self.label2Idx.items()}
            for i in range(len(y_pad)):
                for j in range(len(y_pad[i])):
                    y_pad[i][j] = id2label[y_pad[i][j]]
                pred.append(y_pad[i][:lengths[i]])
            return _x, _y, pred

    @staticmethod
    def load(model_path, device='cpu'):
        model_params = torch.load(model_path, map_location=lambda storage, loc: storage)
        if not os.path.exists(os.path.join(os.path.dirname(model_path), 'params')):
            raise Exception('params data error')

        params_path = os.path.join(os.path.dirname(model_path), 'params')
        params = torch.load(params_path, map_location=lambda storage, loc: storage)
        config = params['config']
        word2Idx = params['word2Idx']
        embedding = params['embedding']
        description = params['description']
        label2Idx = params['label2Idx']
        config.device = device
        model = ConceptTagger(config=config, word2Idx=word2Idx, embedding=embedding,
                                      description=description,
                                      label2Idx=label2Idx)

        model.load_state_dict(model_params['state_dict'])
        return model

    def save(self, path):
        print('save model parameters to [%s]' % path)
        if not os.path.exists(os.path.join(os.path.dirname(path), 'params')):
            params_path = os.path.join(os.path.dirname(path), 'params')
            params = {
                'config': self.config,
                'embedding': self.emb,
                'description': self.description,
                'word2Idx': self.word2Idx,
                'label2Idx': self.label2Idx
            }
            torch.save(params, params_path)
        model_params = {

            'state_dict': self.state_dict()
        }
        torch.save(model_params, path)