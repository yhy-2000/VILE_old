import sys
sys.path += ["./"]
import os
import math
import json
import torch
import pickle
import random
import logging
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset
from typing import List
from transformers import DataCollatorWithPadding,RobertaTokenizer
from typing import Union, List, Tuple, Dict,Any
from dataclasses import dataclass
logger = logging.getLogger(__name__)
import datasets
import pyserini.index.lucene

class TextTokenIdsCache:
    def __init__(self, data_dir, prefix,args):
        meta = json.load(open(f"{data_dir}/{prefix}_meta"))
        self.total_number = meta['total_number']
        self.max_seq_len = meta['embedding_size']
        
        if "query" in data_dir:
            self.ids_arr = np.memmap(
                f"{data_dir}/{prefix}.mmp", 
                shape=(self.total_number, self.max_seq_len), 
                dtype=np.dtype(meta['type']), mode="r")

        else:
            if args.model_s_type=="leaf":
                self.ids_arr = np.memmap(
                    f"{data_dir}/{prefix}.mmp",
                    dtype=np.dtype(meta['type']), 
                    shape=(189679,11,256),
                    mode="r")[:,:args.block_num,:]
                
                self.image_features = np.memmap(
                    f"{data_dir}/image_features.mmp", 
                    dtype=np.dtype("float32"), 
                    mode="r").reshape(-1,11,12,512)[:,:args.block_num,:,:]
                
                assert self.image_features.shape[0]==self.ids_arr.shape[0]==self.total_number
                
            elif args.model_s_type=="pure_text":
                self.ids_arr = np.memmap(
                    f"{data_dir}/{prefix}.mmp", 
                    shape=(189679,11,256),
                    dtype=np.dtype(meta['type']), mode="r")[:,0,:].astype('int64')

        
    def __len__(self):
        return self.total_number
    
    def __getitem__(self, item):
        return self.ids_arr[item]


class SequenceDataset(Dataset):
    def __init__(self, ids_cache, max_seq_length,config):
        self.ids_cache = ids_cache
        self.max_seq_length = max_seq_length
        self.config=config
        
    def __len__(self):  
        return len(self.ids_cache)

    def __getitem__(self, item):
        
        # query
        if self.max_seq_length==32:
            input_ids = torch.tensor(self.ids_cache[item])
        else:
            input_ids = torch.tensor(self.ids_cache[item][0]).unsqueeze(0)

        # print(self.ids_cache[item].shape)
        # print("aaaaaaaaa")
        # print(input_ids.shape)
        attention_mask=torch.zeros_like(input_ids)
        attention_mask[input_ids!=0]=1

        ret_val = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "id": item,
        }
        return ret_val

class MultimodalSequenceDataset(Dataset):
    def __init__(self, ids_cache, config):
        self.ids_cache = ids_cache
        self.max_seq_length = config.max_root_length
        self.config=config
        
    def __len__(self):
            return len(self.ids_cache)


    def add_cls_sep(self,input_ids):
        return [101]+input_ids+[102]

    def __getitem__(self, item):
        if self.config.model_s_type=="leaf":
            input_ids=torch.tensor(self.ids_cache.ids_arr[item])
            attention_mask=torch.zeros_like(input_ids)
            attention_mask[input_ids!=0]=1
            image_features = torch.tensor(self.ids_cache.image_features[item]).reshape(input_ids.shape[0],-1) 
            ret_val={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "image_features": image_features,
                "id": item,
            }
        elif self.config.model_s_type=="pure_text":
            input_ids=torch.tensor(self.ids_cache.ids_arr[item]).unsqueeze(0)
            attention_mask=torch.zeros_like(input_ids)
            attention_mask[input_ids!=0]=1
            image_features = torch.zeros(input_ids.shape[0],12*512)
            ret_val={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "image_features": image_features,
                "id": item,
            }            
        # if self.config.model_s_type=="tree":
        #     ret_val["tree_attention_mask"]=self.text_dataset['tree_attention_mask'][item]
        return ret_val         

class SubsetSeqDataset:
    def __init__(self, subset: List[int], ids_cache, max_seq_length, leaf=False,config=None):
        self.subset = sorted(list(subset))
        if leaf:
            self.alldataset = MultimodalSequenceDataset(ids_cache,config)
        else:
            self.alldataset = SequenceDataset(ids_cache, max_seq_length,config)
        
    def __len__(self):  
        return len(self.subset)

    def __getitem__(self, item):
        return self.alldataset[self.subset[item]]


def load_rel(rel_path):
    reldict = defaultdict(list)
    for line in tqdm(open(rel_path), desc=os.path.split(rel_path)[1]):
        qid, _, pid, _ = line.split()
        qid, pid = int(qid), int(pid)
        reldict[qid].append((pid))
    return dict(reldict)
    

def load_rank(rank_path):
    rankdict = defaultdict(list)
    for line in tqdm(open(rank_path), desc=os.path.split(rank_path)[1]):
        qid, pid, _ = line.split()
        qid, pid = int(qid), int(pid)
        rankdict[qid].append(pid)
    return dict(rankdict)


def pack_tensor_2D(lstlst, default, dtype, length=None):
    batch_size = len(lstlst)
    length = length if length is not None else max(len(l) for l in lstlst)
    tensor = default * torch.ones((batch_size, length), dtype=dtype)
    for i, l in enumerate(lstlst):
        tensor[i, :len(l)] = torch.tensor(l, dtype=dtype)
    return tensor

def pack_tensor_3D(lstlstlst, default, dtype, length=None):
    batch_size = len(lstlstlst)
    group_num = len(lstlstlst[0])
    length = length if length is not None else max(len(lst) for lstlst in lstlstlst for lst in lstlst)
    tensor = default * torch.ones((batch_size, group_num, length), dtype=dtype)
    for i, lstlst in enumerate(lstlstlst):
        for gid in range(group_num):
            tensor[i,gid,:len(lstlst[gid])] = torch.tensor(lstlst[gid][:length], dtype=dtype)
    return tensor


def get_collate_function(max_seq_length):
    cnt = 0
    def collate_function(batch):
        nonlocal cnt
        length=max_seq_length

        input_ids = [x["input_ids"] for x in batch]
        attention_mask = [x["attention_mask"] for x in batch]
        data = {
            "input_ids": pack_tensor_3D(input_ids, default=1, 
                dtype=torch.int64, length=length),
            "attention_mask": pack_tensor_3D(attention_mask, default=0, 
                dtype=torch.int64, length=length),
        }
        if "image_features" in batch[0]:
            image_features = [x["image_features"] for x in batch]
            data["image_features"]= pack_tensor_3D(image_features, default=0, 
                dtype=torch.float64, length=512*12)

        if "tree_attention_mask" in batch[0]:
            tree_attention_mask = [x["tree_attention_mask"] for x in batch]
            data["tree_attention_mask"]= pack_tensor_3D(tree_attention_mask, default=0, 
                dtype=torch.int64)

        ids = [x['id'] for x in batch]
        return data, ids
    return collate_function  



class TrainInbatchDataset(Dataset):
    def __init__(self, rel_file, queryids_cache, docids_cache, config=None):
        self.query_dataset = SequenceDataset(queryids_cache, config.max_query_length,config)
        self.doc_dataset = MultimodalSequenceDataset(docids_cache,config)
        # self.doc_dataset = SequenceDataset(docids_cache, max_doc_length)
        self.reldict = load_rel(rel_file)
        self.qids = sorted(list(self.reldict.keys()))

    def __len__(self):
        return len(self.qids)

    def __getitem__(self, item):
        qid = self.qids[item]
        pid = random.choice(self.reldict[qid])
        query_data = self.query_dataset[qid]
        passage_data = self.doc_dataset[pid]
        return query_data, passage_data


class TrainInbatchWithHardDataset(TrainInbatchDataset):
    def __init__(self, rel_file, rank_file, queryids_cache, 
            docids_cache, hard_num,config):
        TrainInbatchDataset.__init__(self, 
            rel_file, queryids_cache, docids_cache,config)
        self.rankdict = json.load(open(rank_file))
        assert hard_num > 0
        self.hard_num = hard_num

    def __len__(self):
        return len(self.qids)

    def __getitem__(self, item):
        qid = self.qids[item]
        pid = self.reldict[qid][0]
        query_data = self.query_dataset[qid]
        passage_data = self.doc_dataset[pid]
        if str(qid) in self.rankdict and len(self.rankdict[str(qid)])>1:
            candidate_list=self.rankdict[str(qid)] 
        else:
            candidate_list=list(range(len(self.rankdict)))
     
        if pid in candidate_list:
            candidate_list.remove(pid)

        # sample nagatives that rank topper then gt
        hardpids = random.sample(candidate_list, self.hard_num)
        # hardpids = [candidate_list[0]]
        hard_passage_data = [self.doc_dataset[hardpid] for hardpid in hardpids]
        return query_data, passage_data, hard_passage_data


class TrainInbatchWithRandDataset(TrainInbatchDataset):
    def __init__(self, rel_file, queryids_cache, 
            docids_cache, rand_num,config):
        TrainInbatchDataset.__init__(self, 
            rel_file, queryids_cache, docids_cache,config)
        assert rand_num > 0
        self.rand_num = rand_num

    def __getitem__(self, item):
        qid = self.qids[item]
        pid = random.choice(self.reldict[qid])
        query_data = self.query_dataset[qid]
        passage_data = self.doc_dataset[pid]
        randpids = random.sample(range(len(self.doc_dataset)), self.rand_num)
        rand_passage_data = [self.doc_dataset[randpid] for randpid in randpids]
        return query_data, passage_data, rand_passage_data


def single_get_collate_function(max_seq_length, padding=False):
    cnt = 0
    def collate_function(batch):
        nonlocal cnt
        length = max_seq_length

        input_ids = [x["input_ids"] for x in batch]
        attention_mask = [x["attention_mask"] for x in batch]
        # print("bbbbbbbbbbb")
        # print(np.array(input_ids).shape)
        pack_func= pack_tensor_2D if len(input_ids[0].shape)==1 else pack_tensor_3D
        data = {
            "input_ids": pack_func(input_ids, default=1, 
                dtype=torch.int64, length=length),
            "attention_mask": pack_func(attention_mask, default=0, 
                dtype=torch.int64, length=length),
        }
        if "image_features" in batch[0]:
            image_features = [x["image_features"] for x in batch]
            data["image_features"]= pack_tensor_3D(image_features, default=0, 
                dtype=torch.float64, length=512*12)       

        if "tree_attention_mask" in batch[0]:
            tree_attention_mask = [x["tree_attention_mask"] for x in batch]
            data["tree_attention_mask"]= pack_tensor_3D(tree_attention_mask, default=0, 
                dtype=torch.int64)

        ids = [x['id'] for x in batch]
        return data, ids
    return collate_function  


def dual_get_collate_function(max_query_length, max_doc_length, rel_dict, padding=False):
    query_collate_func = single_get_collate_function(max_query_length, padding)
    doc_collate_func = single_get_collate_function(max_doc_length, padding)

    def collate_function(batch):
        query_data, query_ids = query_collate_func([x[0] for x  in batch])
        doc_data, doc_ids = doc_collate_func([x[1] for x in batch])
        rel_pair_mask = [[1 if docid not in rel_dict[qid] else 0 
            for docid in doc_ids]
            for qid in query_ids]
        input_data = {
            "input_query_ids":query_data['input_ids'],
            "query_attention_mask":query_data['attention_mask'],
            "input_doc_ids":doc_data['input_ids'],
            "doc_attention_mask":doc_data['attention_mask'],
            "doc_image_features":doc_data['image_features'],
            "rel_pair_mask":torch.FloatTensor(rel_pair_mask),
            }

        return input_data
    return collate_function  


def triple_get_collate_function(max_query_length, max_doc_length, rel_dict, padding=False,gpu_num=None):
    query_collate_func = single_get_collate_function(max_query_length, padding)
    doc_collate_func = single_get_collate_function(max_doc_length, padding)

    def collate_function(batch):
        query_data, query_ids = query_collate_func([x[0] for x  in batch])
        doc_data, doc_ids = doc_collate_func([x[1] for x in batch])
        hard_doc_data, hard_doc_ids = doc_collate_func(sum([x[2] for x in batch], []))
        
        # 针对多卡mask掉错误的正负例
        rel_pair_mask=[]
        hard_pair_mask=[]
        per_device_batch_size=len(query_ids)//gpu_num
        for gid in range(gpu_num):
            rel_pair_mask.append([[1 if docid not in rel_dict[qid] else 0 
                for docid in doc_ids[gid*per_device_batch_size:gid*per_device_batch_size+per_device_batch_size]]
                for qid in query_ids[gid*per_device_batch_size:gid*per_device_batch_size+per_device_batch_size]])
            hard_pair_mask.append([[1 if docid not in rel_dict[qid] else 0 
                for docid in hard_doc_ids[gid*per_device_batch_size:gid*per_device_batch_size+per_device_batch_size]]
                for qid in query_ids[gid*per_device_batch_size:gid*per_device_batch_size+per_device_batch_size]])
        
        rel_pair_mask=torch.FloatTensor(rel_pair_mask)
        hard_pair_mask=torch.FloatTensor(hard_pair_mask)

        # last step truncate in case the data number cant be devided by gpu number
        if rel_pair_mask.shape[-1]*gpu_num!=query_data["input_ids"].shape[0]:
            saved_num=rel_pair_mask.shape[-1]*gpu_num
            for k in query_data: query_data[k]=query_data[k][:saved_num]
            query_ids=query_ids[:saved_num]
            for k in doc_data: doc_data[k]=doc_data[k][:saved_num]
            doc_ids=doc_ids[:saved_num]
            for k in doc_data: hard_doc_data[k]=hard_doc_data[k][:saved_num]
            hard_doc_ids=hard_doc_ids[:saved_num]

        query_num = len(query_data['input_ids'])
        hard_num_per_query = len(batch[0][2])
        input_data = {
            "input_query_ids":query_data['input_ids'],
            "query_attention_mask":query_data['attention_mask'],
            "input_doc_ids":doc_data['input_ids'],
            "doc_attention_mask":doc_data['attention_mask'],
            "doc_image_features":doc_data['image_features'],
            "other_doc_ids":hard_doc_data['input_ids'].reshape(query_num, hard_num_per_query, -1),
            "other_doc_attention_mask":hard_doc_data['attention_mask'].reshape(query_num, hard_num_per_query, -1),
            "other_doc_image_features":hard_doc_data['image_features'].reshape(query_num, hard_num_per_query, -1),
            "rel_pair_mask": rel_pair_mask,
            "hard_pair_mask":hard_pair_mask,
            }
        if "tree_attention_mask" in doc_data:
            input_data["doc_tree_attention_mask"]=doc_data['tree_attention_mask']
        if "tree_attention_mask" in hard_doc_data:
            input_data["other_doc_tree_attention_mask"]=hard_doc_data['tree_attention_mask'].reshape(query_num, hard_num_per_query,-1)

        return input_data
    return collate_function  

