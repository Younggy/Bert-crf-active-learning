import torch
import json
import os
from torch.utils.data import Dataset

'''
弃用：tag2idx = {"解剖部位": 'A', "手术": 'B', "疾病和诊断": 'C', "影像检查": 'D', "药物": 'E', "实验室检验": 'F'}

labels_tags = {"无"：'O', 解剖部位": 'PAR', "手术": 'SUR', "疾病和诊断": 'DIS', "影像检查": 'IMG', "药物": 'MED', "实验室检验": 'LAB'}
'''

def read_json(filename):
    file = open(filename, 'r', encoding='utf-8')
    entries = []
    for line in file.readlines():
        dic = json.loads(line)
        entries.append(dic)
    return entries


class NerDataset(Dataset):
    def __init__(self, filename, max_len, tokenizer):
        self.max_len = max_len
        self.tokenizer = tokenizer
        entries = read_json(filename)
        sents, tags_li, idx_li = [], [], []
        for k, v in entries[0].items():
            words = [s for s in v['words']]
            tags = v['tags'].split(" ")
            idx_li.append(k)
            sents.append(''.join(words))
            tags_li.append(tags)
        self.tag2id = {"O": 0,"[PAD]":1,\
                       "B-PAR": 2, "I-PAR": 3, \
                       "B-SUR": 4, "I-SUR": 5, \
                       "B-DIS": 6, "I-DIS": 7, \
                       "B-IMG": 8, "I-IMG": 9,  \
                       "B-MED": 10, "I-MED": 11,  \
                       "B-LAB": 12, "I-LAB": 13, \
                       }
        self.id2tag = {0: 'O', 1: '[PAD]',\
                         2: 'B-PAR',3: 'I-PAR',\
                         4: 'B-SUR',5: 'I-SUR',\
                         6: 'B-DIS',7: 'I-DIS',\
                         8: 'B-IMG',9: 'I-IMG',\
                         10: 'B-MED',11: 'I-MED',\
                         12: 'B-LAB',13: 'I-LAB'}
        self.sents, self.tags_li, self.idx_li = sents, tags_li, idx_li

    def __getitem__(self, idx):
        index, words, tags = self.idx_li[idx], self.sents[idx], self.tags_li[idx]

        inputs = {'input_ids': [self.tokenizer.convert_tokens_to_ids('[CLS]')], 'token_type_ids': [0]}
        for w in words[:self.max_len - 2]:
            inputs['input_ids'].append(self.tokenizer.convert_tokens_to_ids(w))
            inputs['token_type_ids'].append(0)
        inputs['input_ids'].append(self.tokenizer.convert_tokens_to_ids('[SEP]'))
        inputs['token_type_ids'].append(0)

        ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]

        cut_tags = ["[PAD]"] + tags[:self.max_len - 2] + ["[PAD]"]
        targets = [self.tag2id[t] for t in cut_tags]

        padding_len = self.max_len - len(ids)

        ids = ids + [0] * padding_len
        token_type_ids = token_type_ids + [0] * padding_len
        targets = targets + [0] * padding_len

        words = " ".join(self.tokenizer.convert_ids_to_tokens(ids))
        tags = " ".join([self.id2tag[t] for t in targets])

        return {
            "index": index,
            "words": words,
            "ids": torch.tensor(ids, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "tags": tags,
            "targets": torch.tensor(targets, dtype=torch.long),
        }

    def __len__(self):
        return len(self.sents)

# class CheckpointManager:
#     def __init__(self, model_dir):
#         self._model_dir = model_dir
#
#     def save_checkpoint(self, state, filename):
#         torch.save(state, os.path.join(self._model_dir, filename))
#
#     def load_checkpoint(self, filename):
#         state = torch.load(os.path.join(self._model_dir, filename), map_location=torch.device('cpu'))
#         return state

class SummaryManager:
    def __init__(self, model_dir):

        self._model_dir = model_dir
        self._summary = {}

    def save(self, filename):
        with open(os.path.join(self._model_dir, filename), mode='w') as io:
            json.dump(self._summary, io, indent=4)

    def load(self, filename):
        with open(os.path.join(self._model_dir, filename), mode='r') as io:
            metric = json.loads(io.read())
        self.update(metric)

    def update(self, summary):
        self._summary.update(summary)

    def reset(self):
        self._summary = {}

    @property
    def summary(self):
        return self._summary