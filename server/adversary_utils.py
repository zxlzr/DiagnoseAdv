import sys
import os
import re
import json
import logging
import OpenAttack
import numpy as np
import opennre
import torch
import argparse
from opennre.model import SoftmaxNN
from OpenAttack.utils.dataset import Dataset, DataInstance
from tqdm import tqdm
import logging
import tensorflow as tf


class REClassifier(OpenAttack.Classifier):
    def __init__(self, model, rel2id, id2rel, device):
        self.model = model
        self.id2rel = id2rel
        self.tokenizer = self.model.sentence_encoder.tokenizer
        self.device = device

        # Store current sample's information in advance
        self.current_entities = []
        self.current_label = -1

    def infer(self, sample):
        item = self.model.sentence_encoder.tokenize(sample)
        item = (i.to(self.device) for i in item)
        logits = self.model.forward(*item)
        logits = self.model.softmax(logits)
        score, pred = logits.max(-1)
        return self.id2rel[pred.item()], score.item()

    def get_prob(self, input_):
        ret = []
        correct_answer = np.zeros(len(self.id2rel))
        correct_answer[self.current_label] = 1.0
        for sent in input_:
            sent = sent.lower()
            valid = True
            # Sanity check 1: make sure the locating tokens are still there
            for special in ['unused0', 'unused1', 'unused2', 'unused3']:
                if sent.count(special) != 1:
                    valid = False
                    break
            # Sanity check 2: make sure the entities are still there
            if valid:
                ents = get_entities(sent)
                valid = ents == self.current_entities
            # Ignore sentences whose special tokens are not valid!
            if not valid:
                ret.append(correct_answer)
                continue
            # Convert data instance to sample
            sample = data2sample(DataInstance(x=sent, y=self.current_label),
                                 self.id2rel)
            # Predict sample label
            items = self.model.sentence_encoder.tokenize(sample)
            items = (i.to(self.device) for i in items)
            with torch.no_grad():
                logits = self.model.forward(*items)
                logits = self.model.softmax(logits).squeeze(0).cpu().numpy()
            ret.append(logits)

        return np.array(ret)
    
    def get_grad(self, input_, labels):
        import pdb; pdb.set_trace()
        ret, grads = [], []
        correct_answer = np.zeros(len(self.id2rel))
        correct_answer[self.current_label] = 1.0
        for sent in input_:
            sent = sent.lower()
            valid = True
            # Sanity check 1: make sure the locating tokens are still there
            for special in ['unused0', 'unused1', 'unused2', 'unused3']:
                if sent.count(special) != 1:
                    valid = False
                    break
            # Sanity check 2: make sure the entities are still there
            if valid:
                ents = get_entities(sent)
                valid = ents == self.current_entities
            # Ignore sentences whose special tokens are not valid!
            if not valid:
                ret.append(correct_answer)
                grads.append(np.zeros(128))
                continue
            # Convert data instance to sample
            sample = data2sample(DataInstance(x=sent, y=self.current_label),
                                 self.id2rel)
            # Predict sample label
            gradient_storage = []
            sentence_encoder = self.model.sentence_encoder
            # grad_input is a tuple containing one tensor (batch, seq_len, model_dim)
            # grad_output is a tuple containing one tensor (batch, seq_len, model_dim)
            gradient_hook = sentence_encoder.bert.embeddings.register_backward_hook(
                lambda module, grad_input, grad_output:
                gradient_storage.append(
                    grad_input[0].detach().squeeze(0).cpu().numpy()))  # (128, 768)

            items = sentence_encoder.tokenize(sample)
            items = (i.to(self.device) for i in items)
            logits = self.model.forward(*items)
            logits = self.model.softmax(logits).squeeze(0).cpu().numpy()
            gradient_hook.remove()  # Remember to remove hook
            ret.append(logits)
            # Process gradient
            gradient_tokens = gradient_storage[-1]
            # What if the length is different?
            grads.append(gradient_tokens)

        return np.array(ret)


def get_entities(sent):
    ent1 = sent[sent.index('unused0') + 7:sent.index('unused1')
                ].lower().split(' ')
    ent1 = list(filter(bool, ent1))
    ent2 = sent[sent.index('unused2') + 7:sent.index('unused3')
                ].lower().split(' ')
    ent2 = list(filter(bool, ent2))
    return [ent1, ent2]


def sample2data(sample, rel2id):
    # Convert a single sample to a DataInstance
    # Process the sentence by adding indicating tokens to head / tail tokens
    tokens = sample['token']
    h_pos, t_pos = sample['h']['pos'], sample['t']['pos']
    head, tail = ' '.join(tokens[h_pos[0]:h_pos[1]]), ' '.join(
        tokens[t_pos[0]:t_pos[1]])
    rev = h_pos[0] > t_pos[0]
    if rev:
        # sent1, tail, sent2, head, sent3
        sent1 = ' '.join(tokens[:t_pos[0]])
        sent2 = ' '.join(tokens[t_pos[1]:h_pos[0]])
        sent3 = ' '.join(tokens[h_pos[1]:])
        sent = ' '.join((sent1, 'unused2', tail, 'unused3',
                         sent2, 'unused0', head, 'unused1', sent3))
    else:
        # sent1, head, sent2, head, sent3
        sent1 = ' '.join(tokens[:h_pos[0]])
        sent2 = ' '.join(tokens[h_pos[1]:t_pos[0]])
        sent3 = ' '.join(tokens[t_pos[1]:])
        sent = ' '.join((sent1, 'unused0', head, 'unused1',
                         sent2, 'unused2', tail, 'unused3', sent3))
    # Remove '_' chars
    sent = re.sub('_', '', sent)
    return DataInstance(x=sent,
                        y=rel2id[sample['relation']])


def data2sample(data, id2rel):
    sent, rel = data.x.lower(), id2rel[data.y]
    # Convert into a legal sample
    pos0, pos1, pos2, pos3 = [sent.find(w) for w in
                              ['unused0', 'unused1', 'unused2', 'unused3']]
    rev = pos0 > pos2
    h, t = sent[pos0 + len('unused0'):pos1], sent[pos2 + len('unused2'):pos3]
    if rev:
        s1, s2 = sent[:pos2], sent[pos3 + len('unused3'):pos0]
        s3 = sent[pos1 + len('unused1'):]
    else:
        s1, s2 = sent[:pos0], sent[pos1 + len('unused1'):pos2]
        s3 = sent[pos3 + len('unused3'):]
    # Convert string to token ids
    h, t, s1, s2, s3 = [part.strip().split()
                        for part in [h, t, s1, s2, s3]]
    if rev:
        words = s1 + t + s2 + h + s3
        h_pos = [len(s1) + len(t) + len(s2)]
        h_pos.append(h_pos[0] + len(h))
        t_pos = [len(s1)]
        t_pos.append(t_pos[0] + len(t))
    else:
        words = s1 + h + s2 + t + s3
        h_pos = [len(s1)]
        h_pos.append(h_pos[0] + len(h))
        t_pos = [len(s1) + len(h) + len(s2)]
        t_pos.append(t_pos[0] + len(t))
    return {'token': words, 'h': {'pos': h_pos}, 't': {'pos': t_pos}, 'relation': rel}


def sample2dataset(sample_list, rel2id):
    # Convert list of samples to dataset object
    data_list = []

    for sample in sample_list:
        data = sample2data(sample, rel2id)
        data_list.append(data)
    dataset = Dataset(data_list=data_list)

    return dataset


def dataset2sample(dataset, id2rel):
    # Convert dataset object to list of samples
    sample_list = []

    for data in dataset:
        sample = data2sample(data, id2rel)
        sample_list.append(sample)

    return sample_list


def diff(origin, modify):
    # Find out changes from origin to modify with Longest-Common-Sequence algorithm
    # Returns 2 action lists: -1 indicates deleted / inserted, while other numbers indicate corresponding position
    n1, n2 = len(origin), len(modify)
    m = [[0 for x in range(n2+1)] for y in range(n1+1)]
    d = [['' for x in range(n2+1)] for y in range(n1+1)]
    for i in range(n1):
        for j in range(n2):
            if origin[i] == modify[j]:
                m[i+1][j+1] = m[i][j] + 1
                d[i+1][j+1] = 'ok'
            elif m[i+1][j] >= m[i][j+1]:
                m[i+1][j+1] = m[i+1][j]
                d[i+1][j+1] = 'j'
            else:
                m[i+1][j+1] = m[i][j+1]
                d[i+1][j+1] = 'i'
    i, j = n1, n2
    actions_origin = [-1 for _ in range(n1)]  # -1 means deletion
    actions_modify = [-1 for _ in range(n2)]  # -1 means insertion
    while i > 0 or j > 0:
        if d[i][j] == 'ok':
            actions_origin[i - 1] = j - 1
            actions_modify[j - 1] = i - 1
            i -= 1
            j -= 1
        elif d[i][j] == 'i':
            i -= 1
        else:
            j -= 1

    return actions_origin, actions_modify


if __name__ == "__main__":
    # Test diff function
    s1 = ['a', 'b', 'c', 'f']
    s2 = ['a', 'c', 'd', 'e', 'f']
    print(diff(s1, s2))

    # Test sample-data convert functions
    sample1 = {"token": ["Tom", "Thabane", "resigned", "in", "October", "last", "year", "to", "form", "the", "All", "Basotho", "Convention", "-LRB-", "ABC", "-RRB-", ",", "crossing", "the", "floor", "with", "17", "members", "of", "parliament",
                         ",", "causing", "constitutional", "monarch", "King", "Letsie", "III", "to", "dissolve", "parliament", "and", "call", "the", "snap", "election", "."], "h": {"pos": [10, 13]}, "t": {"pos": [0, 2]}, "relation": "org:founded_by"}
    rel2id = json.load(open('../dataset/tacred/rel2id.json', 'r'))
    id2rel = {v: k for k, v in rel2id.items()}
    data = sample2data(sample1, rel2id)
    sample2 = data2sample(data, id2rel)
    print(sample1)
    print(data.x)
    print(sample2)
    print(get_entities(data.x))
