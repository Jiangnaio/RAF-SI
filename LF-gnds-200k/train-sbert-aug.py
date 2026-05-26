import os
import random
import json
import torch
from torch import nn
from torch import Tensor
import copy
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup, AutoConfig

from torch.optim import AdamW
from tqdm import tqdm
import gc
import time
import shutil
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
from scipy.sparse import csr_matrix
from xclib.utils.dense import compute_centroid
from xclib.utils.clustering import cluster_balance, b_kmeans_dense
import logging
warnings.filterwarnings("ignore")

# ==================== 日志配置 ====================
def setup_logger(output_dir):
    log_file = os.path.join(output_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logger = logging.getLogger("XMC_Trainer")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# ==================== 配置 ====================
CONFIG = {
    "seed": 42,
    "model_name": "/media/4t/2026/elmo-main/XMC/GND-Subject-test-arctic_m_v2/epoch-3",
    "max_length": 512,
    "batch_size": 32,
    "start_hard_neg_sampling_epoch": 1,
    "gradient_accumulation_steps": 1,
    "num_epochs": 100,
    "learning_rate": 5e-5,
    "warmup_ratio": 0.1,
    "output_dir": "./XMC/GND-Subject-test-arctic1-epoch-3",
    "dataset_dir": "Datasets/GND-Subject-test",
    "train_file": "trn.json",
    "eval_file": "tst.json",
    "test_filter_file": 'filter_labels_test.txt',
    "label_pool_file": "lbl.json",
    "debug_mode": False,
    "train_samples_limit_rate": 0.3,
    "debug_size": 50,
    "max_keep_checkpoints": 5,
    "save_epochs": 5,
    "bf16": True,
    "use_fv": True,
    "loss_type": "decoupled_softmax",
    "per_sample_pos_max": 6,
    "num_neg_samples": 32*6,
    "temperature": 0.05,
    "eps": 1e-8,
    "use_instances_cluster": False,
    "start_cluster_epoch": 2,
    "num_label_clusters": 1024,
    # 新增配置
    "train_encoder": False,           # 是否训练编码器（默认不训练）
    "precompute_embeddings": True,    # 是否预计算嵌入向量（默认开启）
}

# 保存配置及初始化日志
if not os.path.exists(CONFIG["output_dir"]):
    os.makedirs(CONFIG["output_dir"])
shutil.copy(__file__, os.path.join(CONFIG["output_dir"], os.path.basename(__file__)))

with open(os.path.join(CONFIG["output_dir"], 'config.json'), 'w', encoding='utf-8') as f:
    json.dump(CONFIG, f, ensure_ascii=False, indent=4)

logger = setup_logger(CONFIG["output_dir"])

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_raw_embeddings(model, input_ids, attention_mask):
    """获取模型原始输出并进行mean pooling，不进行归一化"""
    return get_embeddings(model, input_ids, attention_mask)

def get_embeddings(model, input_ids, attention_mask, dtype=None, normalize=True):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    if 'arctic' in CONFIG["model_name"]:
        embeddings = outputs[0][:, 0]
    else:
        embeddings = mean_pooling(outputs, attention_mask)
    if normalize:
        return F.normalize(embeddings, p=2, dim=-1)
    return embeddings.float()

# ==================== 数据集 ====================
class XMCDataset(Dataset):
    def __init__(self, data_file, debug_mode=False, debug_size=50, max_pos=-1, for_train=False):
        self.max_pos = max_pos
        examples = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                item=json.loads(line)
                query= item['title'] if 'titles' in data_file.lower() else item['title']+': '+item['content']
                examples.append({
                    "query": query[:CONFIG["max_length"]*10],
                    "pos_ind": item["target_ind"]
                })
        if debug_mode:
            examples = examples[:debug_size]
        if for_train:
            self.examples = examples[:int(len(examples)*CONFIG["train_samples_limit_rate"])]
        self.examples = examples
        assert len(examples) > 0, f"Empty dataset: {data_file}"
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        return {
            "query": example['query'],
            "rand_pos_ind": example["pos_ind"] if self.max_pos < 0 or len(example["pos_ind"]) <= self.max_pos else random.sample(example["pos_ind"], self.max_pos),
            "all_pos_ind": example["pos_ind"],
            "idx":idx
        }

class ClusteringIndex(object):
    def __init__(self, num_instances, num_clusters, num_threads, curr_steps):
        self.num_instances = num_instances
        self.num_clusters = num_clusters
        self.num_threads = num_threads
        self.index = None 
        self.curr_steps = curr_steps 
        self.avg_size = 1
        self.random_clustering()
        
    def random_clustering(self):
        self.index = []
        for i in range(self.num_instances):
           self.index.append([i])
           
    def update(self, X, num_clusters=None):
        assert self.num_instances == len(X)
        _nc = self.num_clusters if num_clusters is None else num_clusters
        self.index, _ = cluster_balance(
            X=X.astype('float32'), 
            clusters=[np.arange(len(X), dtype='int')],
            num_clusters=_nc,
            splitter=b_kmeans_dense,
            num_threads=self.num_threads,
            verbose=True)
        self.avg_size = np.mean(list(map(len, self.index)))
        
    def indices_permutation(self):
        clusters = self.index
        np.random.shuffle(clusters)
        indices = []
        for item in clusters:
            indices.extend(item)
        return np.array(indices) 

class MySampler(torch.utils.data.Sampler[int]):
    def __init__(self, order):
        self.order = order.copy()

    def update_order(self, x):
        self.order[:] = x[:]

    def __iter__(self):
        return iter(self.order)

    def __len__(self) -> int:
        return len(self.order)
    
# ==================== Collator (基础版) ====================
class XMCCollator:
    def __init__(self, tokenizer, label_texts, max_length=32, num_neg_samples=20, hard_negatives=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_texts = label_texts
        self.num_labels = len(label_texts)
        self.label_pool_set = set(range(self.num_labels))
        self.num_neg_samples = num_neg_samples
        self.hard_negatives = hard_negatives if hard_negatives is not None else {}

    def __call__(self, batch):
        queries = [item["query"] for item in batch]
        B = len(queries)

        all_pos_lists = [item["all_pos_ind"] for item in batch]
        rand_pos_lists = [item["rand_pos_ind"] for item in batch]

        cand_pos_set = set()
        for rp in rand_pos_lists:
            cand_pos_set.update(rp)
        
        all_pos_set = set()
        for ap in all_pos_lists:
            all_pos_set.update(ap)
        
        neg_candidates = list(self.label_pool_set - all_pos_set)
        num_neg = min(len(neg_candidates), self.num_neg_samples)
        random_neg_indices = random.sample(neg_candidates, num_neg) if num_neg > 0 else []
        
        batch_indices = [item["idx"] for item in batch] 
        for b_idx in batch_indices:
            hard_neg = self.hard_negatives.get(b_idx, -1)
            if hard_neg != -1:
                random_neg_indices.append(hard_neg)
        random_neg_indices=list(set(random_neg_indices))
        
        cand_inds = list(cand_pos_set) + random_neg_indices
        C = len(cand_inds)
        
        target = torch.zeros((B, C), dtype=torch.float32)
        if C > 0:
            cand_to_col = {ind: idx for idx, ind in enumerate(cand_inds)}
            for i, all_pos in enumerate(all_pos_lists):
                valid_cols = [cand_to_col[p] for p in all_pos if p in cand_to_col]
                if valid_cols:
                    target[i, valid_cols] = 1.0
        
        query_enc = self.tokenizer(queries, max_length=self.max_length, padding=True, truncation=True, return_tensors="pt")
        cand_texts = [self.label_texts[i] for i in cand_inds]
        text_enc = self.tokenizer(cand_texts, max_length=self.max_length, padding=True, truncation=True, return_tensors="pt")
        
        return {
            "query_input_ids": query_enc["input_ids"],
            "query_attention_mask": query_enc["attention_mask"],
            "text_input_ids": text_enc["input_ids"],
            "text_attention_mask": text_enc["attention_mask"],
            "target": target,
            "cand_inds":torch.LongTensor(cand_inds),
            "batch_size": B,
            "num_candidates": C
        }

# ==================== Collator (支持预计算嵌入版) ====================
class XMCCollatorWithCache:
    """支持预计算嵌入的Collator，训练时直接从缓存读取嵌入"""
    def __init__(self, tokenizer, label_texts, label_embs_cache, query_embs_cache,
                 max_length=32, num_neg_samples=20, hard_negatives=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_texts = label_texts
        self.label_embs_cache = label_embs_cache  # [num_labels, D] - 未normalize的原始嵌入
        self.query_embs_cache = query_embs_cache  # [num_queries, D] - 已normalize的查询嵌入
        self.num_labels = len(label_texts)
        self.label_pool_set = set(range(self.num_labels))
        self.num_neg_samples = num_neg_samples
        self.hard_negatives = hard_negatives if hard_negatives is not None else {}

    def __call__(self, batch):
        queries = [item["query"] for item in batch]
        batch_indices = [item["idx"] for item in batch]
        B = len(queries)

        all_pos_lists = [item["all_pos_ind"] for item in batch]
        rand_pos_lists = [item["rand_pos_ind"] for item in batch]

        cand_pos_set = set()
        for rp in rand_pos_lists:
            cand_pos_set.update(rp)
        
        all_pos_set = set()
        for ap in all_pos_lists:
            all_pos_set.update(ap)
        
        neg_candidates = list(self.label_pool_set - all_pos_set)
        num_neg = min(len(neg_candidates), self.num_neg_samples)
        random_neg_indices = random.sample(neg_candidates, num_neg) if num_neg > 0 else []
        
        for b_idx in batch_indices: 
            hard_neg = self.hard_negatives.get(b_idx, -1)
            if hard_neg != -1:
                random_neg_indices.append(hard_neg)
        random_neg_indices = list(set(random_neg_indices))
        
        cand_inds = list(cand_pos_set) + random_neg_indices
        C = len(cand_inds)
        
        target = torch.zeros((B, C), dtype=torch.float32)
        if C > 0:
            cand_to_col = {ind: idx for idx, ind in enumerate(cand_inds)}
            for i, all_pos in enumerate(all_pos_lists):
                valid_cols = [cand_to_col[p] for p in all_pos if p in cand_to_col]
                if valid_cols:
                    target[i, valid_cols] = 1.0
        
        # 从缓存中获取预计算嵌入
        q_emb_pre = torch.stack([self.query_embs_cache[idx] for idx in batch_indices])  # [B, D]
        t_emb_pre = torch.stack([self.label_embs_cache[ind] for ind in cand_inds])  # [C, D]
        
        # 仍需tokenize用于可能的fallback或debug
        query_enc = self.tokenizer(queries, max_length=self.max_length, padding=True, truncation=True, return_tensors="pt")
        
        return {
            "query_input_ids": query_enc["input_ids"],
            "query_attention_mask": query_enc["attention_mask"],
            "target": target,
            "cand_inds": torch.LongTensor(cand_inds),
            "batch_size": B,
            "num_candidates": C,
            "q_emb_pre": q_emb_pre,
            "t_emb_pre": t_emb_pre,
        }

# ==================== 损失函数 ====================
def decoupled_softmax_loss(query_embeds, text_embeds, target, tau=0.05, eps=1e-8, weighted=False, reduction="mean"):
    sim = torch.einsum('bd,cd->bc', query_embeds, text_embeds) / tau
    pos_mask = target > eps
    num_pos = pos_mask.sum(dim=1)
    
    if num_pos.max() == 0:
        return torch.tensor(0.0, device=sim.device, requires_grad=True)

    sim_pos = sim[pos_mask]
    target_vals = target[pos_mask] if weighted else None

    sim_neg = sim.masked_fill(pos_mask, -1e9)
    log_denom_neg = torch.logsumexp(sim_neg, dim=1)
    log_denom_neg_exp = torch.repeat_interleave(log_denom_neg, num_pos)

    log_prob = sim_pos - torch.logaddexp(sim_pos, log_denom_neg_exp)
    if weighted and target_vals is not None:
        log_prob = log_prob * target_vals

    num_pos_exp = torch.repeat_interleave(num_pos, num_pos)
    loss_per_pos = -log_prob / num_pos_exp.clamp(min=eps)

    loss_per_sample = torch.zeros(sim.size(0), device=sim.device)
    row_indices = torch.nonzero(pos_mask, as_tuple=False)[:, 0]
    loss_per_sample.scatter_add_(0, row_indices, loss_per_pos)
    
    if reduction == "mean":
        return loss_per_sample.mean()
    elif reduction == "sum":
        return loss_per_sample.sum()
    elif reduction == "none":
        return loss_per_sample
    return loss_per_sample.sum()

def compute_rae_xmc_loss(query_embeds, text_embeds, target, tau=0.05, eps=1e-8, weighted=False, reduction="mean"):
    return decoupled_softmax_loss(query_embeds, text_embeds, target, tau, eps, weighted, reduction)

# ==================== 评估 ====================
import math
# def compute_metrics(retrieved_indices, ground_truth_sets, max_k=100):
#     metrics = {k: 0.0 for k in ['P@1','P@3','P@5','R@10','R@50','nDCG@1','nDCG@3','nDCG@5','MRR@10']}
#     n_valid = 0
#     log_inv = [1.0 / math.log2(i + 2) for i in range(max_k + 2)]
    
#     for ret, gt in zip(retrieved_indices, ground_truth_sets):
#         if not gt: continue
#         n_valid += 1
#         rel = [1 if (idx != -1 and idx in gt) else 0 for idx in ret]
        
#         h1, h3, h5 = rel[0], sum(rel[:3]), sum(rel[:5])
#         metrics['P@1'] += h1 / 1.0
#         metrics['P@3'] += h3 / 3.0
#         metrics['P@5'] += h5 / 5.0
        
#         h10, h50 = sum(rel[:10]), sum(rel[:50])
#         metrics['R@10'] += h10 / len(gt)
#         metrics['R@50'] += h50 / len(gt)
        
#         for k, name in [(1,'nDCG@1'), (3,'nDCG@3'), (5,'nDCG@5')]:
#             dcg = sum(rel[i] * log_inv[i] for i in range(k))
#             idcg = sum(log_inv[i] for i in range(min(len(gt), k)))
#             metrics[name] += dcg / idcg if idcg > 0 else 0.0
        
#         for r in range(min(len(ret), 10)):
#             if rel[r] == 1:
#                 metrics['MRR@10'] += 1.0 / (r + 1)
#                 break
                
#     for k in metrics:
#         metrics[k] /= n_valid if n_valid > 0 else 1.0
#     return metrics, n_valid

def compute_metrics(retrieved_indices, ground_truth_sets, max_k=100):
    ks = [1, 3, 5, 10, 20, 30, 50, 100]
    
    metric_names = []
    for k in ks:
        metric_names.extend([f'P@{k}', f'R@{k}', f'F1@{k}', f'nDCG@{k}'])
    metric_names.append('MRR@100')
    
    metrics = {k: 0.0 for k in metric_names}
    n_valid = 0
    log_inv = [1.0 / math.log2(i + 2) for i in range(max_k + 2)]
    
    for ret, gt in zip(retrieved_indices, ground_truth_sets):
        if not gt: continue
        n_valid += 1
        rel = [1 if (idx != -1 and idx in gt) else 0 for idx in ret]
        
        for k in ks:
            # 🔥 关键修改：如果当前 k 超过了实际检索返回的长度，则跳过该 k 值的计算
            if k > len(ret):
                continue
                
            hits = sum(rel[:k])
            
            p_k = hits / k
            metrics[f'P@{k}'] += p_k
            
            r_k = hits / len(gt)
            metrics[f'R@{k}'] += r_k
            
            if p_k + r_k > 0:
                f1_k = 2 * p_k * r_k / (p_k + r_k)
            else:
                f1_k = 0.0
            metrics[f'F1@{k}'] += f1_k
             
            dcg = sum(rel[i] * log_inv[i] for i in range(k))
            idcg = sum(log_inv[i] for i in range(min(len(gt), k)))
            metrics[f'nDCG@{k}'] += dcg / idcg if idcg > 0 else 0.0
        
        # MRR@100 同样需要防御
        mrr_k = min(len(ret), 100)
        for r in range(mrr_k):
            if rel[r] == 1:
                metrics['MRR@100'] += 1.0 / (r + 1)
                break       
                
    for k in metrics:
        metrics[k] /= n_valid if n_valid > 0 else 1.0
    return metrics, n_valid


def gpu_chunked_retrieval(q_embs, l_embs, top_k, q_chunk=1024, l_chunk=20000, device='cuda'):
    if not torch.cuda.is_available(): device = 'cpu'
    N_q, D = q_embs.shape
    N_l, _ = l_embs.shape
    
    final_scores = np.full((N_q, top_k), -1e9, dtype=np.float32)
    final_indices = np.full((N_q, top_k), -1, dtype=np.int64)
    
    for q_s in tqdm(range(0, N_q, q_chunk), desc="🔍 Query Chunks", leave=False):
        q_e = min(q_s + q_chunk, N_q)
        Q = torch.from_numpy(q_embs[q_s:q_e]).float().to(device)
        bs = Q.shape[0]
        
        best_s = torch.full((bs, top_k), -1e9, device=device, dtype=torch.float32)
        best_i = torch.full((bs, top_k), -1, device=device, dtype=torch.int64)
        
        for l_s in range(0, N_l, l_chunk):
            l_e = min(l_s + l_chunk, N_l)
            L = torch.from_numpy(l_embs[l_s:l_e]).float().to(device)
            
            with torch.no_grad():
                sim = torch.matmul(Q, L.t())
            
            k_l = min(top_k, l_e - l_s)
            chunk_s, chunk_i = torch.topk(sim, k=k_l, dim=1, sorted=False)
            chunk_i = chunk_i + l_s
            
            merged_s = torch.cat([best_s, chunk_s], dim=1)
            merged_i = torch.cat([best_i, chunk_i], dim=1)
            best_s, idx_merge = torch.topk(merged_s, k=top_k, dim=1, sorted=True)
            best_i = torch.gather(merged_i, 1, idx_merge)
            del L, sim, chunk_s, chunk_i, merged_s, merged_i, idx_merge
        
        final_scores[q_s:q_e] = best_s.cpu().float().numpy()
        final_indices[q_s:q_e] = best_i.cpu().float().numpy()
        del Q, best_s, best_i
    
    return final_scores, final_indices

def apply_filter_to_topk(indices, filter_file):
    filter_file = os.path.join(CONFIG["dataset_dir"], filter_file)
    if filter_file is None or not os.path.exists(filter_file): return indices
    
    mapping = np.loadtxt(filter_file, dtype=int, ndmin=2)
    if mapping.size == 0: return indices
    
    filter_dict = {}
    for q_idx, l_idx in mapping:
        filter_dict.setdefault(q_idx, set()).add(l_idx)
    
    N_q, K = indices.shape
    filtered = np.full((N_q, K), -1, dtype=indices.dtype)
    for i in range(N_q):
        forbidden = filter_dict.get(i)
        if forbidden is None:
            filtered[i] = indices[i]
        else:
            valid = [idx for idx in indices[i] if idx not in forbidden]
            if valid: filtered[i, :len(valid)] = valid
    return filtered

@torch.no_grad()
def evaluate_with_metrics(augmenter, tokenizer, eval_dataset, label_texts, device, dtype, tau=0.05, top_k=100, test_filter_file=None):
    augmenter.eval()
    logger.info("🏷️ Starting Evaluation...")
    
    label_embs = []
    for i in tqdm(range(0, len(label_texts), CONFIG["batch_size"]), desc="encoding labels", ncols=80, dynamic_ncols=False):
        batch = label_texts[i:i+CONFIG["batch_size"]]
        enc = tokenizer(batch, max_length=CONFIG["max_length"], padding=True, truncation=True, return_tensors="pt").to(device)
        emb = augmenter.encode_label(enc["input_ids"], enc["attention_mask"], torch.arange(i, min(i+CONFIG["batch_size"], len(label_texts))).to(device))
        label_embs.append(emb.cpu())
    label_embs = torch.cat(label_embs, dim=0).float().numpy()
    
    logger.info("❓ Encoding queries...")
    query_texts = [eval_dataset[idx]["query"] for idx in range(len(eval_dataset))]
    gt_sets = [set(eval_dataset[idx]["all_pos_ind"]) for idx in range(len(eval_dataset))]

    query_embs = []
    for i in tqdm(range(0, len(query_texts), CONFIG["batch_size"]), desc="encoding tst", ncols=80, dynamic_ncols=False):
        batch = query_texts[i:i+CONFIG["batch_size"]]
        enc = tokenizer(batch, max_length=CONFIG["max_length"], padding=True, truncation=True, return_tensors="pt").to(device)
        emb = augmenter.encode_text(enc["input_ids"], enc["attention_mask"])
        query_embs.append(emb.cpu())
    query_embs = torch.cat(query_embs, dim=0).float().numpy()
    
    logger.info(f"⚡ Retrieval top-{top_k}...")
    _, indices = gpu_chunked_retrieval(query_embs, label_embs, top_k=top_k, q_chunk=CONFIG["batch_size"], l_chunk=50000, device=device)
    if test_filter_file: indices = apply_filter_to_topk(indices, test_filter_file)
    
    metrics, n_valid = compute_metrics(indices.tolist(), gt_sets, max_k=top_k)
    logger.info(f"📊 Valid queries: {n_valid}/{len(eval_dataset)}")
    logger.info(f"📊 Metrics: {metrics}")
    augmenter.train()
    cleanup_memory()
    return metrics, n_valid

def load_label_texts(label_file, dataset_dir):
    path = os.path.join(dataset_dir, label_file)
    labels = []
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            labels = [json.loads(line)['title'] for line in f.readlines()]
    return list(labels)

def save_model(model, tokenizer, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

def cleanup_memory():
    gc.collect()
    torch.cuda.empty_cache()

# ==================== 预计算嵌入辅助函数 ====================
@torch.no_grad()
def precompute_all_embeddings(model, tokenizer, dataset, label_texts, 
                              device, dtype, batch_size, max_length):
    """预计算所有查询和标签的嵌入向量"""
    logger.info("🔄 Pre-computing all embeddings...")
    
    # 预计算标签嵌入 (不normalize，由augmenter的encode_label处理)
    label_embs = []
    for i in tqdm(range(0, len(label_texts), batch_size), desc="Pre-encoding labels", ncols=80, dynamic_ncols=False):
        batch = label_texts[i:i+batch_size]
        enc = tokenizer(batch, max_length=max_length, padding=True, 
                       truncation=True, return_tensors="pt").to(device)
        emb = get_embeddings(model, enc["input_ids"], enc["attention_mask"], 
                           dtype, normalize=False)
        label_embs.append(emb.cpu())
    label_embs = torch.cat(label_embs, dim=0)
    
    # 预计算查询嵌入 (已normalize)
    query_embs = []
    query_texts = [dataset[idx]["query"] for idx in range(len(dataset))]
    for i in tqdm(range(0, len(query_texts), batch_size), desc="Pre-encoding queries", ncols=80, dynamic_ncols=False):
        batch = query_texts[i:i+batch_size]
        enc = tokenizer(batch, max_length=max_length, padding=True, 
                       truncation=True, return_tensors="pt").to(device)
        emb = get_embeddings(model, enc["input_ids"], enc["attention_mask"], 
                           dtype, normalize=True)
        query_embs.append(emb.cpu())
    query_embs = torch.cat(query_embs, dim=0)
    
    cleanup_memory()
    logger.info(f"✅ Pre-computed: {len(label_embs)} label embeddings, {len(query_embs)} query embeddings")
    return label_embs, query_embs

# ==================== 网络结构 ====================
class SingleLayerTransformerEncoder(nn.Module):
    def __init__(self, d_model=512, n_head=1, dim_feedforward=2048, dropout=0.1, norm_first=False):
        super(SingleLayerTransformerEncoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, n_head, dim_feedforward, dropout, activation='gelu', norm_first=norm_first)

    def forward(self, x, src_key_padding_mask=None):
        x = torch.stack(x)
        if src_key_padding_mask is not None:
            output = self.encoder_layer(x, src_key_padding_mask=src_key_padding_mask)
            valid_mask = (src_key_padding_mask == 0).float().transpose(0, 1).unsqueeze(-1).expand(-1, -1, output.shape[2])
            masked_output = output * valid_mask
            sum_output = torch.sum(masked_output, dim=0)
            valid_counts = torch.sum(valid_mask, dim=0)
            valid_counts = torch.clamp(valid_counts, min=1.0)
            mean_pooled = sum_output / valid_counts
            return mean_pooled
        else:
            output = self.encoder_layer(x)
            return torch.mean(output, 0)

class Augmenter(nn.Module):
    def __init__(self, model, config, label_embs_numpy=None, n_clusters=None, 
                 use_fv=True, train_encoder=False):
        super(Augmenter, self).__init__()
        self.encoder = model
        self.use_fv = use_fv
        self.train_encoder = train_encoder
        self.hidden_size = config.hidden_size
        
        # 如果不需要训练编码器，冻结参数并设为评估模式
        if not self.train_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()
        
        if self.use_fv:
            self.n_clusters = n_clusters if n_clusters is not None else 0
            self.register_buffer("label_mapping", torch.empty(0, dtype=torch.long))
            self.snet_weight = nn.Parameter(torch.Tensor(1, self.hidden_size))
            self.combiner = SingleLayerTransformerEncoder(
                d_model=self.hidden_size,
                n_head=4,
                dim_feedforward=512,
                dropout=0.1,
            )

            if label_embs_numpy is not None and n_clusters is not None:
                logger.info(f"Clustering labels into {n_clusters} groups...")
                clusters, label_mapping = cluster_balance(
                    X=label_embs_numpy.astype('float32'), 
                    clusters=[np.arange(len(label_embs_numpy), dtype='int64')], 
                    num_clusters=n_clusters, 
                    splitter=b_kmeans_dense, 
                    num_threads=4, 
                    verbose=True
                )
                self.n_clusters = len(clusters)
                self.label_mapping = torch.from_numpy(label_mapping).long()
                
                self.snet_weight = nn.Parameter(torch.Tensor(self.n_clusters, self.hidden_size))
                rows, cols = [], []
                for cluster_idx, cluster_array in enumerate(clusters):
                    for element in cluster_array:
                        rows.append(cluster_idx)
                        cols.append(element)
                data = np.ones(len(rows), dtype=np.float32)
                fv_map_gt = csr_matrix((data, (rows, cols)), shape=(self.n_clusters, len(label_embs_numpy)))
                
                centroids = compute_centroid(torch.from_numpy(label_embs_numpy), fv_map_gt.transpose().tocsr(), reduction='mean')
                self.snet_weight.data.copy_(torch.from_numpy(centroids).float())

    def encode_text(self, input_ids, attention_mask):
        if not self.train_encoder:
            with torch.no_grad():
                return get_embeddings(self.encoder, input_ids, attention_mask, normalize=True)
        return get_embeddings(self.encoder, input_ids, attention_mask, normalize=True)

    def encode_label(self, input_ids, attention_mask, lbl_inds):
        if not self.train_encoder:
            with torch.no_grad():
                t_emb = get_embeddings(self.encoder, input_ids, attention_mask, normalize=False)
        else:
            t_emb = get_embeddings(self.encoder, input_ids, attention_mask, normalize=False)
            
        if not self.use_fv:
            return F.normalize(t_emb, p=2, dim=-1)
            
        cls_ids = self.label_mapping[lbl_inds.to(self.snet_weight.device)]
        fv = self.snet_weight[cls_ids].squeeze(1)
        if len(fv.shape) == 1:
            fv = fv.unsqueeze(0)
            
        sequence = [t_emb, fv]
        enhance_t_emb = self.combiner(sequence)
        return F.normalize(enhance_t_emb, p=2, dim=-1)

    def forward(self, q_ids, q_mask, t_ids, t_mask, lbl_inds, 
                q_emb_pre=None, t_emb_pre=None):
        """
        支持预计算嵌入的前向传播
        - q_emb_pre: 预计算的查询嵌入 [B, D] (已normalize)
        - t_emb_pre: 预计算的候选标签嵌入 [C, D] (未normalize)
        """
        if q_emb_pre is not None and not self.train_encoder:
            q_emb = q_emb_pre
        else:
            q_emb = self.encode_text(q_ids, q_mask)
            
        if t_emb_pre is not None and not self.train_encoder:
            t_emb = t_emb_pre
            # 对预计算的标签嵌入进行fv增强和normalize
            if self.use_fv:
                cls_ids = self.label_mapping[lbl_inds.to(self.snet_weight.device)]
                fv = self.snet_weight[cls_ids].squeeze(1)
                if len(fv.shape) == 1:
                    fv = fv.unsqueeze(0)
                sequence = [t_emb, fv]
                t_emb = self.combiner(sequence)
            t_emb = F.normalize(t_emb, p=2, dim=-1)
        else:
            t_emb = self.encode_label(t_ids, t_mask, lbl_inds)
            
        return q_emb, t_emb

    def save_augmenter(self, tokenizer, output_dir):
        timestamp = datetime.now().strftime("%Y%m%d%H")
        save_dir = f"{output_dir}_{timestamp}"
        os.makedirs(save_dir, exist_ok=True)
        
        # self.encoder.save_pretrained(save_dir)#由于该模型没有训练，故而不需要保存
        tokenizer.save_pretrained(save_dir)
        
        augmenter_state = {
            'snet_weight': self.snet_weight.data,
            'combiner': self.combiner.state_dict(),
            'label_mapping': self.label_mapping,
            'n_clusters': self.n_clusters,
            'use_fv': self.use_fv,
            'hidden_size': self.hidden_size,
            'train_encoder': self.train_encoder
        }
        torch.save(augmenter_state, os.path.join(save_dir, "augmenter_state.bin"))
        logger.info(f"✅ Model and augmenter saved to {save_dir}")
        return save_dir

    @classmethod
    def load_augmenter(cls, model_name_or_path, device):
        state_path = os.path.join(model_name_or_path, "augmenter_state.bin")
        if not os.path.exists(state_path):
            raise FileNotFoundError(f"augmenter_state.bin not found in {model_name_or_path}")
            
        augmenter_state = torch.load(state_path, map_location=device, weights_only=False)
        
        config = AutoConfig.from_pretrained(model_name_or_path)
        model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
        
        augmenter = cls(model, config, label_embs_numpy=None, n_clusters=None, 
                       use_fv=augmenter_state['use_fv'],
                       train_encoder=augmenter_state.get('train_encoder', False))
        
        augmenter.n_clusters = augmenter_state['n_clusters']
        augmenter.label_mapping = augmenter_state['label_mapping'].to(device)
        augmenter.snet_weight = nn.Parameter(augmenter_state['snet_weight'].to(device))
        augmenter.combiner.load_state_dict(augmenter_state['combiner'])
        
        augmenter.to(device)
        augmenter.eval()
        return augmenter

# ==================== 主流程 ====================
def main():
    set_seed(CONFIG["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if CONFIG["bf16"] and torch.cuda.is_bf16_supported() else torch.float32
    
    config = AutoConfig.from_pretrained(CONFIG["model_name"], trust_remote_code=True)
    logger.info(f"🔧 Model config hidden_size: {config.hidden_size}")

    for d in [CONFIG["output_dir"], f"{CONFIG['output_dir']}/checkpoints", f"{CONFIG['output_dir']}/plots"]:
        os.makedirs(d, exist_ok=True)
        
    if hasattr(config, "use_memory_efficient_attention"):
        config.use_memory_efficient_attention = False

    logger.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    
    # 加载模型
    if 'arctic' in CONFIG["model_name"]:
        model = AutoModel.from_pretrained(CONFIG["model_name"], config=config, dtype=torch.bfloat16, add_pooling_layer=False, low_cpu_mem_usage=True, trust_remote_code=True)    
    elif 'qwen' in CONFIG["model_name"]:
        model = AutoModel.from_pretrained(CONFIG["model_name"], attn_implementation="flash_attention_2", dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True)
    else:
        model = AutoModel.from_pretrained(CONFIG["model_name"], torch_dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True)
        
    model = model.to(device)
    if CONFIG.get("use_gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    
    label_texts = load_label_texts(CONFIG["label_pool_file"], CONFIG["dataset_dir"])
    logger.info(f"Label pool size: {len(label_texts)}")
    
    train_dataset = XMCDataset(os.path.join(CONFIG["dataset_dir"], CONFIG["train_file"]), 
                              CONFIG["debug_mode"], 10000, max_pos=CONFIG['per_sample_pos_max'], for_train=True)
    eval_dataset = XMCDataset(os.path.join(CONFIG["dataset_dir"], CONFIG["eval_file"]), 
                             CONFIG["debug_mode"], 200, CONFIG['per_sample_pos_max'])
    
    # ========== 预计算嵌入 ==========
    if CONFIG["precompute_embeddings"]:
        logger.info("🔄 Pre-computing embeddings for training optimization...")
        label_embs_cache, query_embs_cache = precompute_all_embeddings(
            model, tokenizer, train_dataset, label_texts, 
            device, dtype, CONFIG["batch_size"], CONFIG["max_length"]
        )
        label_embs_numpy = label_embs_cache.float().numpy()
    else:
        label_embs_numpy = []
        with torch.no_grad():
            for i in tqdm(range(0, len(label_texts), CONFIG["batch_size"]), desc="Pre-encoding labels", ncols=80, dynamic_ncols=False):
                batch = label_texts[i:i+CONFIG["batch_size"]]
                enc = tokenizer(batch, max_length=CONFIG["max_length"], padding=True, truncation=True, return_tensors="pt").to(device)
                emb = get_embeddings(model, enc["input_ids"], enc["attention_mask"], dtype)
                label_embs_numpy.append(emb.cpu())
        label_embs_numpy = torch.cat(label_embs_numpy, dim=0).float().numpy()
        label_embs_cache = query_embs_cache = None
    
    # ========== 创建augmenter ==========
    n_clusters = CONFIG.get("num_label_clusters", len(label_texts)//100)
    augmenter = Augmenter(model, config, label_embs_numpy, n_clusters, 
                         use_fv=CONFIG["use_fv"], 
                         train_encoder=CONFIG.get("train_encoder", False)).to(device)
    
    del label_embs_numpy
    cleanup_memory()

    # ========== 优化器: 仅训练fv和combiner ==========
    gp = []
    if augmenter.use_fv:
        gp += [{"params": augmenter.snet_weight, "lr": 10*CONFIG["learning_rate"], "name": "snet_weight"}]
        gp += [{"params": augmenter.combiner.parameters(), "lr": 10*CONFIG["learning_rate"], "name": "combiner"}]
        logger.info(f"🎯 Training params: snet_weight({augmenter.snet_weight.shape}), combiner params: {sum(p.numel() for p in augmenter.combiner.parameters())}")
    else:
        logger.warning("⚠️ use_fv is False, no trainable params!")
        
    optimizer = AdamW(gp, lr=CONFIG["learning_rate"])
    
    # ========== 创建DataLoader ==========
    if CONFIG["precompute_embeddings"]:
        collator = XMCCollatorWithCache(tokenizer, label_texts, label_embs_cache, query_embs_cache,
                                       CONFIG["max_length"], CONFIG["num_neg_samples"])
    else:
        collator = XMCCollator(tokenizer, label_texts, CONFIG["max_length"], CONFIG["num_neg_samples"])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, 
                             collate_fn=collator, pin_memory=True, num_workers=2)
    
    total_steps = len(train_loader) // CONFIG["gradient_accumulation_steps"] * CONFIG["num_epochs"]
    warmup_steps = int(total_steps * CONFIG["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    global_step, best_eval_loss = 0, float('inf')
    tau = CONFIG["temperature"]
    
    logger.info(f"🚀 Start training: {total_steps} steps, warmup: {warmup_steps}")
    logger.info(f"🔒 Encoder frozen: {not CONFIG.get('train_encoder', False)}, Precomputed embeddings: {CONFIG['precompute_embeddings']}")
    
    for epoch in range(CONFIG["num_epochs"]):      
        # Hard negative sampling
        if epoch >= CONFIG["start_hard_neg_sampling_epoch"]:
            augmenter.eval()
            logger.info("Sampling hard negatives...")
            with torch.no_grad(): 
                # 获取增强后的标签嵌入用于top-k检索
                if CONFIG["precompute_embeddings"]:
                    label_embs_aug = []
                    for i in tqdm(range(0, len(label_texts), CONFIG["batch_size"]), desc="Augment label embeddings", ncols=80, dynamic_ncols=False):
                        lbl_inds = torch.arange(i, min(i+CONFIG["batch_size"], len(label_texts))).to(device)
                        raw_emb = label_embs_cache[i:min(i+CONFIG["batch_size"], len(label_texts))].to(device)
                        if augmenter.use_fv:
                            cls_ids = augmenter.label_mapping[lbl_inds]
                            fv = augmenter.snet_weight[cls_ids].squeeze(1)
                            if len(fv.shape) == 1:
                                fv = fv.unsqueeze(0)
                            sequence = [raw_emb, fv]
                            enhanced = augmenter.combiner(sequence)
                            # # 简单组合: (raw + fv) / 2, 实际应根据combiner逻辑调整
                            # enhanced = (raw_emb + fv) / 2
                            label_embs_aug.append(F.normalize(enhanced, p=2, dim=-1).cpu())
                        else:
                            label_embs_aug.append(F.normalize(raw_emb, p=2, dim=-1).cpu())
                    label_embs = torch.cat(label_embs_aug, dim=0).to(device)
                else:
                    label_embs = []
                    for i in tqdm(range(0, len(label_texts), CONFIG["batch_size"]), desc="Encode label", ncols=80, dynamic_ncols=False):
                        batch = label_texts[i:i+CONFIG["batch_size"]]
                        enc = tokenizer(batch, max_length=CONFIG["max_length"], padding=True, truncation=True, return_tensors="pt").to(device)
                        lbl_inds = torch.arange(i, min(i+CONFIG["batch_size"], len(label_texts))).to(device)
                        emb = augmenter.encode_label(enc["input_ids"], enc["attention_mask"], lbl_inds)
                        label_embs.append(emb) 
                    label_embs = torch.cat(label_embs, dim=0)
                
                # 查询top-k
                train_query_texts = [train_dataset[idx]["query"] for idx in range(len(train_dataset))]
                trn_topk_ids = []
                for i in tqdm(range(0, len(train_query_texts), CONFIG["batch_size"]), desc="Query & Topk", ncols=80, dynamic_ncols=False):
                    batch_idx = list(range(i, min(i+CONFIG["batch_size"], len(train_query_texts))))
                    if CONFIG["precompute_embeddings"]:
                        emb = query_embs_cache[batch_idx].to(device)
                    else:
                        batch = train_query_texts[i:i+CONFIG["batch_size"]]
                        enc = tokenizer(batch, max_length=CONFIG["max_length"], padding=True, truncation=True, return_tensors="pt").to(device)
                        emb = augmenter.encode_text(enc["input_ids"], enc["attention_mask"])
                    
                    sim = torch.matmul(emb.float(), label_embs.t())
                    _, chunk_i = torch.topk(sim, k=100, dim=1, sorted=True)
                    trn_topk_ids.append(chunk_i.cpu())
                trn_topk_ids = torch.cat(trn_topk_ids, dim=0)
                
                hard_negatives = {}
                for idx in tqdm(range(len(train_dataset)), desc="Sample hard neg", ncols=80, dynamic_ncols=False):
                    gt = set(train_dataset[idx]["all_pos_ind"])
                    pred = trn_topk_ids[idx].tolist()
                    candidates = list(set(pred) - gt)
                    if candidates:
                        hard_negatives[idx] = random.choice(candidates)
                collator.hard_negatives = hard_negatives
                
            del label_embs, trn_topk_ids, train_query_texts # enc, emb, batch, sim, chunk_i, 
            cleanup_memory()
            
        # ========== 训练循环 ==========
        augmenter.train()
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}", ncols=80, dynamic_ncols=False)
        avg_loss = 0.0
        
        for step, batch in enumerate(progress):
            q_ids = batch["query_input_ids"].to(device, non_blocking=True)
            q_mask = batch["query_attention_mask"].to(device, non_blocking=True)
            l_ind = batch["cand_inds"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            
            # 获取预计算嵌入
            q_emb_pre = batch.get("q_emb_pre", None)
            t_emb_pre = batch.get("t_emb_pre", None)
            if q_emb_pre is not None:
                q_emb_pre = q_emb_pre.to(device, non_blocking=True)
            if t_emb_pre is not None:
                t_emb_pre = t_emb_pre.to(device, non_blocking=True)
            
            with torch.autocast(device_type="cuda", dtype=dtype):
                q_emb, t_emb = augmenter(q_ids, q_mask, None, None, l_ind, 
                                        q_emb_pre=q_emb_pre, t_emb_pre=t_emb_pre)
                
                if CONFIG["loss_type"] == "decoupled_softmax":
                    loss = decoupled_softmax_loss(q_emb, t_emb, target, tau=tau, eps=CONFIG["eps"])
                elif CONFIG["loss_type"] == "rae-xmc":
                    loss = compute_rae_xmc_loss(q_emb, t_emb, target, tau=tau)
         
            loss.backward()
            # 只clip可训练参数
            torch.nn.utils.clip_grad_norm_([p for p in augmenter.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            avg_loss += loss.item() 
            global_step += 1
            progress.set_postfix({"loss": avg_loss/(progress.n+1), "lr": scheduler.get_last_lr()[0]})
            
            if step % 10 == 0:
                cleanup_memory()

        epoch_avg_loss = avg_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} finished. Average Loss: {epoch_avg_loss:.4f}")

        if (epoch+1) % CONFIG["save_epochs"]==0:
            augmenter.save_augmenter(tokenizer, f"{CONFIG['output_dir']}/epoch-{epoch+1}")
            evaluate_with_metrics(augmenter, tokenizer, eval_dataset, label_texts, device, dtype, 
                            tau=0.05, top_k=101, test_filter_file=CONFIG["test_filter_file"]) 
            
    augmenter.save_augmenter(tokenizer, f"{CONFIG['output_dir']}/final")
    logger.info("✅ Training completed.")


if __name__ == "__main__":
    main()
