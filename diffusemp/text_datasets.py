# import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader, Dataset

import torch, os
import json, pickle
import psutil
import datasets
from datasets import Dataset as Dataset2
from tqdm import tqdm, trange

def load_data_text(
    batch_size, 
    seq_len, 
    deterministic=False, 
    data_args=None, 
    model_emb=None,
    split='train', 
    loaded_vocab=None,
    loop=True,
):
    """
    For a dataset, create a generator over (seqs, kwargs) pairs.

    Each seq is an (bsz, len, h) float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for some meta information.

    :param batch_size: the batch size of each returned pair.
    :param seq_len: the max sequence length (one-side).
    :param deterministic: if True, yield results in a deterministic order.
    :param data_args: including dataset directory, num of dataset, basic settings, etc.
    :param model_emb: loaded word embeddings.
    :param loaded_vocab: loaded word vocabs.
    :param loop: loop to get batch data or not.
    """

    print('#'*30, '\nLoading text data...')

    training_data = get_corpus(data_args, seq_len, split=split, loaded_vocab=loaded_vocab)

    dataset = TextDataset(
        training_data,
        data_args,
        model_emb=model_emb
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,  # 20,
        # drop_last=True,
        # shuffle=not deterministic,
        shuffle=False,
        num_workers=0, debug
    )
    if loop:
        return infinite_loader(data_loader)
    else:
        # print(data_loader)
        return iter(data_loader)

def infinite_loader(data_loader):
    while True:
        yield from data_loader

def helper_tokenize(sentence_lst, vocab_dict, seq_len, use_visible_mask=False, split='train'):
    # Process.memory_info is expressed in bytes, so convert to megabytes
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    raw_datasets = Dataset2.from_dict(sentence_lst)
    print(raw_datasets)
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    def tokenize_function(examples): tokenize function
        if not use_visible_mask:
            input_id_x = vocab_dict.encode_token(examples['src'])
            input_id_y = vocab_dict.encode_token(examples['trg'])
            result_dict = {'input_id_x': input_id_x, 'input_id_y': input_id_y}
            # def encode_token_with_control(self, context, target, emotion, ert, ext, ipt, it, sf): #
        else:
            x, y, mask, length = [], [], [], [] 
            for i in trange(len(examples['src'])):
                input_id_x, input_id_y, visible_mask, context_length = vocab_dict.encode_token_with_control(
                    examples['src'][i], examples['trg'][i], examples['emo'][i], examples['er'][i], examples['ex'][i], examples['ip'][i], examples['it'][i], examples['sf'][i])
                x.append(input_id_x)
                y.append(input_id_y)
                mask.append(visible_mask)
                length.append(context_length)
            result_dict = {'input_id_x': x, 'input_id_y': y, "visible_mask" :mask, "context_len":length}
        return result_dict

    if use_visible_mask:
        # vocab_dict.add_special_tokens_()
        print('vocab dict size: ', vocab_dict.vocab_size)
        tokenized_path = f'/path-to-repo/DiffusEmp/datasets/EmpatheticDialogue/mask-fine/{split}_{seq_len}_tokenized_datasets.pkl'
    if use_visible_mask and os.path.exists(tokenized_path):
        with open(tokenized_path, 'rb') as f:
            tokenized_datasets = pickle.load(f)
    else:
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=1,
            remove_columns=['src', 'trg'] if not use_visible_mask else ['src', 'trg', 'emo', 'er', 'ex', 'ip', 'it', 'sf'],
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )
        if use_visible_mask:
            with open (tokenized_path,'wb') as f:
                pickle.dump(tokenized_datasets, f)
    print('### tokenized_datasets', tokenized_datasets)
    print('### tokenized_datasets...example', tokenized_datasets['input_id_x'][0])
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    def merge_and_mask(group_lst):
        lst = []
        mask = []
        for i in range(len(group_lst['input_id_x'])):
            end_token = group_lst['input_id_x'][i][-1]
            src = group_lst['input_id_x'][i][:-1]
            trg = group_lst['input_id_y'][i][:-1]
            while len(src) + len(trg) > seq_len - 3:
                if len(src)>len(trg):
                    src.pop()
                elif len(src)<len(trg):
                    trg.pop()
                else:
                    src.pop()
                    trg.pop()
            src.append(end_token)
            trg.append(end_token)

            lst.append(src + [vocab_dict.sep_token_id] + trg)
            mask.append([0]*(len(src)+1))
        group_lst['input_ids'] = lst
        group_lst['input_mask'] = mask
        return group_lst
    
    def ed_merge_and_mask(group_lst):
        lst = []
        mask = []
        vm_ = []
        for i in range(len(group_lst['input_id_x'])):
            end_token = group_lst['input_id_x'][i][-1]
            src = group_lst['input_id_x'][i][:(group_lst["context_len"][i]-1)]
            ctr = group_lst['input_id_x'][i][group_lst["context_len"][i]:-1]
            trg = group_lst['input_id_y'][i][:-1]
            vm = group_lst['visible_mask'][i]
            while len(src)+ len(ctr) + len(trg) > seq_len - 3:
                if len(src) > 0:
                    src.pop()
                    vm.pop(len(src)-1)
                    for j in range(len(vm)):
                        vm[j].pop(len(src)-1)
                else:
                    trg.pop()
                    vm.pop(len(trg)-1)
                    for j in range(len(vm)):
                        vm[j].pop(len(trg)-1)
                # if len(src)>len(trg):
                #     src.pop()
                # elif len(src)<len(trg):
                #     trg.pop()
                # else:
                #     src.pop()
                #     trg.pop()
            src.append(end_token)
            # ctr.append(end_token)
            trg.append(end_token)

            lst.append(src + [vocab_dict.sep_token_id] + ctr + trg)
            mask.append([0]*(len(src)+1+len(ctr)))
            vm_.append(vm)
        group_lst['input_ids'] = lst
        group_lst['input_mask'] = mask
        group_lst['visible_mask'] = vm_
        return group_lst
    
    merge_path = f'/path-to-repo/DiffusEmp/datasets/EmpatheticDialogue/mask-fine/{split}_{seq_len}_merge_tokenized_datasets.pkl'
    if use_visible_mask and os.path.exists(merge_path):
        with open (merge_path,'rb') as f:
            tokenized_datasets = pickle.load(f)
    else:
        tokenized_datasets = tokenized_datasets.map(
            merge_and_mask if not use_visible_mask else ed_merge_and_mask,
            batched=True,
            num_proc=1,
            desc=f"merge and mask",
        )
        if use_visible_mask:
            with open (merge_path,'wb') as f:
                pickle.dump(tokenized_datasets, f)

    def pad_function(group_lst):
        max_length = seq_len
        group_lst['input_ids'] = _collate_batch_helper(group_lst['input_ids'], vocab_dict.pad_token_id, max_length)
        group_lst['input_mask'] = _collate_batch_helper(group_lst['input_mask'], 1, max_length)
        if use_visible_mask:
            group_lst['visible_mask'] = _collate_batch_helper_vm(group_lst['visible_mask'], 1, max_length)
        return group_lst

    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    pad_path = f'x/path-to-repo/DiffusEmp/datasets/EmpatheticDialogue/mask-fine/{split}_{seq_len}_pad_tokenized_datasets.pkl'
    if use_visible_mask and os.path.exists(pad_path):
        with open (pad_path,'rb') as f:
            lm_datasets = pickle.load(f)
        # lm_datasets={'input_ids':[], 'input_mask':[], 'visible_mask':[]}
        # for n in trange(41):
        #     path = f'/path-to-repo/DiffusEmp/datasets/EmpatheticDialogue/mask-fine/pkls/pad_lm_datasets_{n}.pkl'
        #     with open (path,'rb') as f:
        #         temp = pickle.load(f)
        #         lm_datasets['input_ids'].extend(temp['input_ids'])
        #         lm_datasets['input_mask'].extend(temp['input_mask'])
        #         lm_datasets['visible_mask'].extend(temp['visible_mask'])
        #         print(f"{n}.1 RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
        #         del temp
        #         print(f"{n}.2 RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
        #     print(f"{n}.3 RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    else:
        input_ids = tokenized_datasets['input_ids']
        vms = tokenized_datasets['visible_mask']
        lm_datasets = tokenized_datasets.map(
            pad_function,
            remove_columns=['input_id_x', 'input_id_y', 'context_len'] if use_visible_mask else [],
            batched=True,
            num_proc=1,
            load_from_cache_file=False,
            desc=f"padding",
        )
        if use_visible_mask:
            with open (pad_path,'wb') as f:
                pickle.dump(lm_datasets, f)

    # print(lm_datasets[0], 'padded dataset')
    # lm_datasets = Dataset2.from_dict(lm_datasets)
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    raw_datasets = datasets.DatasetDict()
    raw_datasets['train'] = lm_datasets
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    return raw_datasets


def get_corpus(data_args, seq_len, split='train', loaded_vocab=None):

    print('#'*30, '\nLoading dataset {} from {}...'.format(data_args.dataset, data_args.data_dir))

    sentence_lst = {'src':[], 'trg': []}
    if data_args.use_visible_mask:
        sentence_lst = {'src':[], 'trg': [], "emo":[], "er":[], "ex":[], "ip":[], "it":[], "sf":[]}
    
    if split == 'train':
        print('### Loading form the TRAIN set...')
        path = f'{data_args.data_dir}/train.jsonl'
    elif split == 'valid':
        print('### Loading form the VALID set...')
        path = f'{data_args.data_dir}/valid.jsonl'
    elif split == 'test':
        print('### Loading form the TEST set...')
        path = f'{data_args.data_dir}/test.jsonl'
    else:
        assert False, "invalid split for dataset"

    with open(path, 'r') as f_reader:
        if not data_args.use_visible_mask:
            for row in f_reader:
                sentence_lst['src'].append(json.loads(row)['src'].strip())
                sentence_lst['trg'].append(json.loads(row)['trg'].strip())
        else:
            for row in f_reader:
                d = json.loads(row)
                sentence_lst['src'].append(d['src'].strip())
                sentence_lst['trg'].append(d['trg'])
                sentence_lst['emo'].append(d['emo'].strip())#mask tesemotion
                sentence_lst['er'].append(d['ert'].strip())
                sentence_lst['ex'].append(d['ext'].strip())
                sentence_lst['ip'].append(d['ipt'].strip())
                sentence_lst['it'].append(d['it'].strip())
                sentence_lst['sf'].append(d['sf'].strip())


    print('### Data samples...\n', sentence_lst['src'][:2], sentence_lst['trg'][:2])
        
    # get tokenizer.
    vocab_dict = loaded_vocab

    train_dataset = helper_tokenize(sentence_lst, vocab_dict, seq_len, data_args.use_visible_mask, split)
    return train_dataset


class TextDataset(Dataset):
    def __init__(self, text_datasets, data_args, model_emb=None):
        super().__init__()
        self.text_datasets = text_datasets
        self.length = len(self.text_datasets['train'])
        self.data_args = data_args
        self.model_emb = model_emb

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with torch.no_grad():

            input_ids = self.text_datasets['train'][idx]['input_ids']
            hidden_state = self.model_emb(torch.tensor(input_ids))

            # obtain the input vectors, only used when word embedding is fixed (not trained end-to-end)
            arr = np.array(hidden_state, dtype=np.float32)

            out_kwargs = {}
            out_kwargs['input_ids'] = np.array(self.text_datasets['train'][idx]['input_ids'])
            out_kwargs['input_mask'] = np.array(self.text_datasets['train'][idx]['input_mask'])
            if self.data_args.use_visible_mask:
                out_kwargs['visible_mask'] = np.array(self.text_datasets['train'][idx]['visible_mask'])

            return arr, out_kwargs

def _collate_batch_helper(examples, pad_token_id, max_length, return_mask=False):
    result = torch.full([len(examples), max_length], pad_token_id, dtype=torch.int64).tolist()
    mask_ = torch.full([len(examples), max_length], pad_token_id, dtype=torch.int64).tolist()
    for i, example in enumerate(examples):
        curr_len = min(len(example), max_length)
        result[i][:curr_len] = example[:curr_len]
        mask_[i][:curr_len] = [1] * curr_len
    if return_mask:
        return result, mask_
    return result

def _collate_batch_helper_vm(examples, pad_token_id, max_length, return_mask=False):
    vm_ = torch.full([len(examples), max_length, max_length], pad_token_id, dtype=torch.int16).tolist()
    for i, example in enumerate(examples):
        curr_len = min(len(example), max_length)
        for j in range(curr_len):
            vm_[i][j][:curr_len] = example[j][:curr_len]
    return vm_
