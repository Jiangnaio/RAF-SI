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
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
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
warnings.filterwarnings("ignore")

# ==================== 配置 ====================
CONFIG = {
    "seed": 42,
    "model_name": "Snowflake/snowflake-arctic-embed-m-v2.0", 
    "max_length": 512,
    "batch_size": 32,
    "start_hard_neg_sampling_epoch": 1,
    "gradient_accumulation_steps": 1,
    "momentum":0.99, # SGD momentum，动量更新的一个弊端是更新速度较慢，标签无法高效提供梯度下降方向的信号。
    "num_epochs": 3,
    "learning_rate": 5e-5, #在大batch_size下，可以考虑较大的学习率
    "warmup_ratio": 0.1,
    "output_dir": "./XMC/GND-Subject-test-arctic_m_v2",
    "dataset_dir": "Datasets/GND-Subject-test",
    "train_file": "trn.json",
    "eval_file": "tst.json",
    "test_filter_file": 'filter_labels_test.txt',  # 可选：过滤文件路径
    "label_pool_file": "lbl.json",  # 全局标签池文件
    "debug_mode": False,
    "debug_size": 50,
    "max_keep_checkpoints": 10,
    "save_epochs": 1,
    "bf16": True,
    "per_sample_pos_max": 8,
    "num_neg_samples": 32*6,          # 每个query采样的负标签数量
    "temperature": 0.05,
    "eps": 1e-8,
}

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

def get_embeddings(model, input_ids, attention_mask, dtype=None):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask,return_dict=True)
    # embeddings = mean_pooling(outputs, attention_mask)
    embeddings = outputs[0][:, 0]
    # embeddings = last_token_pool(outputs.last_hidden_state, attention_mask)
    return F.normalize(embeddings, p=2, dim=-1)

def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

# ==================== 数据集 ====================
class XMCDataset(Dataset):
    """只存储query和正标签文本，负标签在collator中从全局池采样"""
    def __init__(self, data_file, debug_mode=False, debug_size=50, max_pos=-1, for_train=False):
        self.max_pos = max_pos
        examples = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                item=json.loads(line)
                query= item['title'] if 'titles' in data_file.lower() else item['title']+': '+item['content']
                examples.append({
                    "query": query,
                    "pos_ind": item["target_ind"]  # List[str]
                })
        if debug_mode:
            examples = examples[:debug_size]
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

class XMCCollator:
    """
    """
    def __init__(self, tokenizer, label_texts, max_length=32, num_neg_samples=20, hard_negatives=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_texts = label_texts
        self.num_labels = len(label_texts)
        # 🔹 预计算标签池集合，避免重复转换
        self.label_pool_set = set(range(self.num_labels))
        self.num_neg_samples = num_neg_samples
        self.hard_negatives = hard_negatives if hard_negatives is not None else {}

    def __call__(self, batch):
        queries = [item["query"] for item in batch]
        B = len(queries)

        # 🔹 预提取列表，避免循环内重复访问
        all_pos_lists = [item["all_pos_ind"] for item in batch]   # 用于构建 target
        rand_pos_lists = [item["rand_pos_ind"] for item in batch] # 用于构建 cand_inds

        # ==================== 1. 构建 cand_inds ====================
        # 1.1 收集所有 rand_pos_ind 并去重 (作为候选正标签)
        cand_pos_set = set()
        for rp in rand_pos_lists:
            cand_pos_set.update(rp)
        
        # 1.2 收集所有 all_pos_ind 用于负采样排除 (确保负样本不与任何正标签冲突)
        all_pos_set = set()
        for ap in all_pos_lists:
            all_pos_set.update(ap)
        
        # 1.3 高效负采样: 集合差集 + random.sample
        neg_candidates = list(self.label_pool_set - all_pos_set)
        num_neg = min(len(neg_candidates), self.num_neg_samples)
        random_neg_indices = random.sample(neg_candidates, num_neg) if num_neg > 0 else []
        #添加硬负样本
        batch_indices = [item["idx"] for item in batch] 
        for b_idx in batch_indices:
            hard_neg = self.hard_negatives.get(b_idx, -1)
            if hard_neg != -1:
                random_neg_indices.append(hard_neg)
        #负样本去重        
        random_neg_indices=list(set(random_neg_indices))
        
        # 1.4 组合候选索引: 先正后负 (顺序不影响，但保持一致性)
        cand_inds = list(cand_pos_set) + random_neg_indices
   
        C = len(cand_inds)
        
        # ==================== 2. 构建 target 矩阵 (🔥 核心优化) ====================
        target = torch.zeros((B, C), dtype=torch.float32)
        
        if C > 0:
            # 🔹 关键优化: 建立 全局标签索引 → cand_inds列索引 的 O(1) 映射
            cand_to_col = {ind: idx for idx, ind in enumerate(cand_inds)}
            
            # 🔹 向量化填充: 对每个样本，批量查找其 all_pos_ind 在 cand_inds 中的列位置
            for i, all_pos in enumerate(all_pos_lists):
                # 只标记那些同时在 cand_inds 中的正标签 (理论上都应该在，但防御性编程)
                valid_cols = [cand_to_col[p] for p in all_pos if p in cand_to_col]
                if valid_cols:
                    target[i, valid_cols] = 1.0
        
        # ==================== 3. Tokenize ====================
        query_enc = self.tokenizer(
            queries, 
            max_length=self.max_length, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        )
        
        cand_texts = [self.label_texts[i] for i in cand_inds]
        text_enc = self.tokenizer(
            cand_texts, 
            max_length=self.max_length, 
            padding=True,
            truncation=True, 
            return_tensors="pt"
        )
        
        return {
            "query_input_ids": query_enc["input_ids"],
            "query_attention_mask": query_enc["attention_mask"],
            "text_input_ids": text_enc["input_ids"],
            "text_attention_mask": text_enc["attention_mask"],
            "target": target,          # [B, C] float matrix
            "batch_size": B,
            "num_candidates": C
        }
    
# ==================== 损失函数：解耦Softmax ====================
def decoupled_softmax_loss(query_embeds, text_embeds, target, tau=0.05, eps=1e-8, 
                           weighted=False, reduction="mean"):
    """
    Args:
        query_embeds: [B, D]
        text_embeds: [C, D]  # 全局候选标签embedding
        target: [B, C]  # 0/1 float matrix
        tau: temperature
        eps: small value for stability
        weighted: whether to use target values as weights
        reduction: "mean" | "sum" | "none"
    Returns:
        scalar loss (or [B] if reduction="none")
    """
    # 1. 计算相似度 [B, C]
    sim = torch.einsum('bd,cd->bc', query_embeds, text_embeds) / tau
    
    # 2. 定位正样本
    pos_mask = target > eps  # [B, C]
    num_pos = pos_mask.sum(dim=1)  # [B]
    
    if num_pos.max() == 0:
        return torch.tensor(0.0, device=sim.device, requires_grad=True)

    # 3. 提取正样本相似度与权重
    sim_pos = sim[pos_mask]  # [total_pos]
    target_vals = target[pos_mask] if weighted else None

    # 4. 负样本 logsumexp（正样本位置置-100隔离）
    sim_neg = sim.masked_fill(pos_mask, -1e9)
    log_denom_neg = torch.logsumexp(sim_neg, dim=1)  # [B]

    # 5. 将负样本分母广播至每个正样本位置
    log_denom_neg_exp = torch.repeat_interleave(log_denom_neg, num_pos)

    # 6. 解耦对数概率计算
    # log(p) = s_pos/τ - logaddexp(s_pos/τ, log∑exp(s_neg/τ))
    log_prob = sim_pos - torch.logaddexp(sim_pos, log_denom_neg_exp)
 
    if weighted and target_vals is not None:
        log_prob = log_prob * target_vals

    # 7. 按样本内正样本数量归一化
    num_pos_exp = torch.repeat_interleave(num_pos, num_pos)
    loss_per_pos = -log_prob / num_pos_exp.clamp(min=eps)

    # 8. 聚合到样本级别
    loss_per_sample = torch.zeros(sim.size(0), device=sim.device)
    row_indices = torch.nonzero(pos_mask, as_tuple=False)[:, 0]
    loss_per_sample.scatter_add_(0, row_indices, loss_per_pos)
    
    # 9. Reduction
    if reduction == "mean":
        return loss_per_sample.mean()
    elif reduction == "sum":
        return loss_per_sample.sum()
    elif reduction == "none":
        return loss_per_sample
    return loss_per_sample.sum()

# ==================== 评估 ====================
import math
def compute_metrics(retrieved_indices, ground_truth_sets, max_k=100):
    """
    计算XMC标准评估指标
    Args:
        retrieved_indices: List[List[int]], shape (N_q, top_k)
        ground_truth_sets: List[Set[int]], 每个query的正标签索引集合
    Returns:
        dict of metrics
    """
    metrics = {k: 0.0 for k in ['P@1','P@3','P@5','R@10','R@50','nDCG@1','nDCG@3','nDCG@5','MRR@10']}
    n_valid = 0
    log_inv = [1.0 / math.log2(i + 2) for i in range(max_k + 2)]
    
    for ret, gt in zip(retrieved_indices, ground_truth_sets):
        if not gt:
            continue
        n_valid += 1
        
        # 标记相关项: 跳过-1(过滤占位符)
        rel = [1 if (idx != -1 and idx in gt) else 0 for idx in ret]
        
        # Precision@K
        h1, h3, h5 = rel[0], sum(rel[:3]), sum(rel[:5])
        metrics['P@1'] += h1 / 1.0
        metrics['P@3'] += h3 / 3.0
        metrics['P@5'] += h5 / 5.0
        
        # Recall@K
        h10, h50 = sum(rel[:10]), sum(rel[:50])
        metrics['R@10'] += h10 / len(gt)
        metrics['R@50'] += h50 / len(gt)
        
        # nDCG@K
        for k, name in [(1,'nDCG@1'), (3,'nDCG@3'), (5,'nDCG@5')]:
            dcg = sum(rel[i] * log_inv[i] for i in range(k))
            idcg = sum(log_inv[i] for i in range(min(len(gt), k)))
            metrics[name] += dcg / idcg if idcg > 0 else 0.0
        
        # MRR@10
        for r in range(min(len(ret), 10)):
            if rel[r] == 1:
                metrics['MRR@10'] += 1.0 / (r + 1)
                break
    
    # 平均
    for k in metrics:
        metrics[k] /= n_valid if n_valid > 0 else 1.0
    return metrics, n_valid

def gpu_chunked_retrieval(q_embs, l_embs, top_k, q_chunk=1024, l_chunk=20000, device='cuda'):
    """GPU分块检索，避免显存溢出"""
    if not torch.cuda.is_available():
        device = 'cpu'
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
    """适配 (N_q, top_k) 结构的标签过滤"""
    filter_file = os.path.join(CONFIG["dataset_dir"], filter_file)
    if filter_file is None or not os.path.exists(filter_file):
        return indices
    
    print(f"📥 Loading filter: {filter_file}")
    mapping = np.loadtxt(filter_file, dtype=int, ndmin=2)
    if mapping.size == 0:
        return indices
    
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
            if valid:
                filtered[i, :len(valid)] = valid
    return filtered

# ==================== 完整评估函数（含指标计算） ====================
@torch.no_grad()
def evaluate_with_metrics(model,tokenizer, eval_dataset, label_texts, device, dtype, 
                          tau=0.05, top_k=100, test_filter_file=None):
    """
    完整评估流程: 编码->检索->过滤->计算指标
    """
    model.eval()
    print("🏷️ Evaluate...")
    label_embs = []
    for i in tqdm(range(0, len(label_texts), CONFIG["batch_size"]), desc="encoding labels"):
        batch = label_texts[i:i+CONFIG["batch_size"]]
        enc = tokenizer(batch, max_length=CONFIG["max_length"], padding=True,
                       truncation=True, return_tensors="pt").to(device)
        emb = get_embeddings(model, enc["input_ids"], enc["attention_mask"], dtype)
        label_embs.append(emb.cpu())
    label_embs = torch.cat(label_embs, dim=0).float().numpy()  # [N_labels, D]
    
    print("❓ Encoding queries...")
    query_texts = [eval_dataset[idx]["query"] for idx in range(len(eval_dataset))]
    gt_sets = [set(eval_dataset[idx]["all_pos_ind"]) for idx in range(len(eval_dataset))]

    
    query_embs = []
    for i in tqdm(range(0, len(query_texts), CONFIG["batch_size"]), desc="encoding tst"):
        batch = query_texts[i:i+CONFIG["batch_size"]]
        enc = tokenizer(batch, max_length=CONFIG["max_length"], padding=True,
                       truncation=True, return_tensors="pt").to(device)
        emb = get_embeddings(model, enc["input_ids"], enc["attention_mask"], dtype)
        query_embs.append(emb.cpu())
    query_embs = torch.cat(query_embs, dim=0).float().numpy()  # [N_queries, D]
    
    print(f"⚡ Retrieval top-{top_k}...")
    _, indices = gpu_chunked_retrieval(
        query_embs, label_embs, top_k=top_k,
        q_chunk=CONFIG["batch_size"], l_chunk=50000, device=device
    )
    
    if test_filter_file:
        indices = apply_filter_to_topk(indices, test_filter_file)
    
    print("📊 Computing metrics...")
    metrics, n_valid = compute_metrics(indices.tolist(), gt_sets, max_k=top_k)
    print(f"📊 Valid queries: {n_valid}/{len(eval_dataset)}")
    print(f"📊 {metrics}")
    model.train()
    cleanup_memory()
    return metrics, n_valid


# ==================== 工具函数 ====================
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



# ==================== 主流程 ====================
def main():
    set_seed(CONFIG["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if CONFIG["bf16"] and torch.cuda.is_bf16_supported() else torch.float32
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(CONFIG["model_name"], trust_remote_code=True)
    print(f"🔧 Model config hidden_size: {config.hidden_size}")  # 确保是 768

    # 创建目录
    for d in [CONFIG["output_dir"], f"{CONFIG['output_dir']}/checkpoints", f"{CONFIG['output_dir']}/plots"]:
        os.makedirs(d, exist_ok=True)
    # 1. 加载 config
    config = AutoConfig.from_pretrained(
        CONFIG["model_name"],
        trust_remote_code=True
    )
    
    # 2. 关闭 memory_efficient_attention
    if hasattr(config, "use_memory_efficient_attention"):
        config.use_memory_efficient_attention = False

    # 加载tokenizer和模型
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    if 'arctic' in CONFIG["model_name"]:
        model = AutoModel.from_pretrained(
            CONFIG["model_name"], 
            config=config,
            dtype=torch.bfloat16,
            add_pooling_layer=False,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )    
    elif 'qwen' in CONFIG["model_name"]:
        model = AutoModel.from_pretrained(
            CONFIG["model_name"],
            attn_implementation="flash_attention_2",
            dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    else:
        model = AutoModel.from_pretrained(
            CONFIG["model_name"],
            # torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        
    # model = AutoModel.from_pretrained(CONFIG["model_name"], torch_dtype=dtype, trust_remote_code=True)
    model = model.to(device)
    if CONFIG.get("use_gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    # #复制模型副本
    # model_copy = copy.deepcopy(model)
    # model_copy = model_copy.to(device)
    # # 将副本模型的所有参数设置为不可训练
    # for param in model_copy.parameters():
    #     param.requires_grad = False
    
    # 加载标签池和数据集
    label_texts = load_label_texts(CONFIG["label_pool_file"], CONFIG["dataset_dir"])
    print(f"Label pool size: {len(label_texts)}")
    
    train_dataset = XMCDataset(
        os.path.join(CONFIG["dataset_dir"], CONFIG["train_file"]),
        CONFIG["debug_mode"], 10000, max_pos=CONFIG['per_sample_pos_max'],for_train=True)
    eval_dataset = XMCDataset(
        os.path.join(CONFIG["dataset_dir"], CONFIG["eval_file"]), 
        CONFIG["debug_mode"], 200, CONFIG['per_sample_pos_max'])
    
    collator = XMCCollator(tokenizer, label_texts, CONFIG["max_length"], CONFIG["num_neg_samples"])
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, 
                              collate_fn=collator, pin_memory=True, num_workers=2)
    # 不需要使用eval_loader，因为评估时直接使用数据集，而且使用数据加载器会出错
    # 优化器和调度器
    total_steps = len(train_loader) // CONFIG["gradient_accumulation_steps"] * CONFIG["num_epochs"]
    warmup_steps = int(total_steps * CONFIG["warmup_ratio"])
    optimizer = AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    # 训练循环
    global_step, best_eval_loss = 0, float('inf')
    tau = CONFIG["temperature"]
    
    print(f"Start training: {total_steps} steps, warmup: {warmup_steps}")
    
    for epoch in range(CONFIG["num_epochs"]):
        if epoch >= CONFIG["start_hard_neg_sampling_epoch"]:
            model.eval()
            print("sampling hard nagtive")
            with torch.no_grad(): 
                # 编码所有标签
                label_embs = []
                for i in tqdm(range(0, len(label_texts), CONFIG["batch_size"]),desc="encode label"):
                    batch = label_texts[i:i+CONFIG["batch_size"]]
                    enc = tokenizer(batch, max_length=CONFIG["max_length"], padding=True,
                                    truncation=True, return_tensors="pt").to(device)
                    emb = get_embeddings(model, enc["input_ids"], enc["attention_mask"], dtype)
                    label_embs.append(emb)#bf16格式
                label_embs = torch.cat(label_embs, dim=0)  # [C, D]
                train_query_texts = [train_dataset[idx]["query"] for idx in range(len(train_dataset))]
                trn_topk_ids = []
                for i in tqdm(range(0, len(train_query_texts), CONFIG["batch_size"]),desc="encode train query and compute topk"):
                    batch = train_query_texts[i:i+CONFIG["batch_size"]]
                    enc = tokenizer(batch, max_length=CONFIG["max_length"], padding=True,
                                    truncation=True, return_tensors="pt").to(device)
                    emb = get_embeddings(model, enc["input_ids"], enc["attention_mask"], dtype)
                    # 直接计算分数
                    sim = torch.matmul(emb, label_embs.t())
                    _, chunk_i = torch.topk(sim, k=100, dim=1, sorted=True)
                    trn_topk_ids.append(chunk_i.cpu())
                trn_topk_ids = torch.cat(trn_topk_ids, dim=0)  # [B, 100]
                hard_negatives = {}
                for idx in tqdm(range(len(train_dataset)),desc="sample hard nagtive"):
                    gt = set(train_dataset[idx]["all_pos_ind"])
                    pred = trn_topk_ids[idx].tolist()
                    hard_negatives[idx] = random.choice(list(set(pred) - gt)) #随机选择一个负样本
                collator.hard_negatives = hard_negatives
        model.train()
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        avg_loss = 0.0
        for step, batch in enumerate(progress):
            q_ids = batch["query_input_ids"].to(device, non_blocking=True)
            q_mask = batch["query_attention_mask"].to(device, non_blocking=True)
            t_ids = batch["text_input_ids"].to(device, non_blocking=True)
            t_mask = batch["text_attention_mask"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            
            with torch.autocast(device_type="cuda", dtype=dtype):
                q_emb = get_embeddings(model, q_ids, q_mask, dtype)  # [B, D]
                # 主题使用副本模型进行编码，但是不计算梯度
                t_emb = get_embeddings(model, t_ids, t_mask, dtype)  # [C, D]
                # with torch.no_grad():
                #     t_emb = get_embeddings(model_copy, t_ids, t_mask, dtype)  # [C, D]
                loss = decoupled_softmax_loss(q_emb, t_emb, target, tau=tau, eps=CONFIG["eps"])
            
         
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            avg_loss += loss.item() 
            global_step += 1
            # # 动量更新副本模型的参数
            # for param, param_copy in zip(model.parameters(), model_copy.parameters()):
            #     param_copy.data = param_copy.data * CONFIG["momentum"] + param.data * (1 - CONFIG["momentum"]) #标签不参与梯度更新，但是参与动量更新，这与只是使用一个模型，但是标签不参与梯度更新的模型有什么区别？
      
            progress.set_postfix({"loss": avg_loss/(progress.n+1), "lr": scheduler.get_last_lr()[0]})
                
            
            # 减少内存清理频率，提高运行速度
            if step % 10 == 0:
                cleanup_memory()

        
        if (epoch+1) % CONFIG["save_epochs"]==0:
            save_model(model, tokenizer, f"{CONFIG['output_dir']}/epoch-{epoch+1}") #先保存，再评估
            evaluate_with_metrics(model,tokenizer, eval_dataset, label_texts, device, dtype, 
                            tau=0.05, top_k=100, test_filter_file=CONFIG["test_filter_file"]) 
            
    
    # Final save
    save_model(model, tokenizer, f"{CONFIG['output_dir']}/final")
    print("✅ Training completed.")

if __name__ == "__main__":
    main()
    # 保存配置
    with open(f"{CONFIG['output_dir']}/config.json", "w") as f:
        json.dump(CONFIG, f, indent=4)
    # 保存本脚本
    with open(f"{CONFIG['output_dir']}/train.py", "w") as f:
        f.write(open(__file__).read())
    
