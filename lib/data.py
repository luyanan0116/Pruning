# lib/data.py
import os
import random
import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer

# 设置随机种子的辅助函数
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 1. 加载本地 WikiText-2 数据集
def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    # 【已修改】更新为你上传并解压的本地路径
    local_wiki_path = "/root/dw2/Lya/Pruning/dataset_wikitext-raw"
    print(f"Loading local WikiText-2 from raw text files in: {local_wiki_path}")
    
    # 确定解压后的文件路径（正常解压后包含 wiki.train.raw 和 wiki.test.raw）
    train_file = os.path.join(local_wiki_path, "wiki.train.raw")
    test_file = os.path.join(local_wiki_path, "wiki.test.raw")
    
    # 兼容性处理：如果解压时自动多创建了一层 "wikitext-2-raw" 文件夹
    if not os.path.exists(train_file):
        train_file = os.path.join(local_wiki_path, "wikitext-2-raw", "wiki.train.raw")
        test_file = os.path.join(local_wiki_path, "wikitext-2-raw", "wiki.test.raw")
    
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"找不到 wiki.train.raw 文件，请检查路径。当前尝试过的路径:\n1. {os.path.join(local_wiki_path, 'wiki.train.raw')}\n2. {train_file}")

    # 【已修改】使用 'text' 模式加载本地的 .raw 纯文本文件
    traindata = load_dataset('text', data_files={'train': train_file}, split='train')
    testdata = load_dataset('text', data_files={'test': test_file}, split='test')

    # 后续分词与采样逻辑保持完全不变
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    set_seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

# 2. 加载本地 C4 数据集
def get_c4(nsamples, seed, seqlen, tokenizer):
    # 本地 C4 数据集路径
    local_c4_path = "/root/dw2/Lya/Pruning/dataset_c4"
    print(f"Loading local C4 from: {local_c4_path}")

    # 自动适配：优先使用 load_from_disk，若不成功则使用 load_dataset 载入本地目录
    if os.path.exists(os.path.join(local_c4_path, "dataset_dict.json")) or os.path.exists(os.path.join(local_c4_path, "state.json")):
        dataset = load_from_disk(local_c4_path)
        traindata = dataset['train']
        valdata = dataset['validation']
    else:
        # 如果是包含未打包 json.gz 文件的本地目录，单独指定格式载入
        if os.path.exists(os.path.join(local_c4_path, "en")):
            traindata = load_dataset('json', data_files={'train': os.path.join(local_c4_path, 'en/c4-train.00000-of-01024.json.gz')}, split='train')
            valdata = load_dataset('json', data_files={'validation': os.path.join(local_c4_path, 'en/c4-validation.00000-of-00008.json.gz')}, split='validation')
        else:
            traindata = load_dataset(local_c4_path, split='train')
            valdata = load_dataset(local_c4_path, split='validation')

    set_seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc

# 3. 统一的数据加载器调度接口
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if 'c4' in name:
        return get_c4(nsamples, seed, seqlen, tokenizer)
    raise ValueError(f"Unknown dataset name: {name}")