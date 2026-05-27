#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
重排序训练与评估脚本
输出目录: Results/{dataset}/rerank/
支持标签增强开关: --use_label_augmentation
python 7_train-rerank.py --model_path /media/4t/2026/elmo-main/XMC/GND-Subject-test-arctic_m_v2/final --dataset_dir Datasets/GND-Subject-test

"""
import os, sys, json, argparse, time, gc, warnings, math, logging
from datetime import datetime
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix, save_npz
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoConfig
import numpy as np
import pickle
import shutil

warnings.filterwarnings('ignore')

# ========================== 日志配置 ==========================
def setup_logger(output_dir):
    """初始化日志记录器"""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"rerank_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logger = logging.getLogger("Rerank_Evaluator")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


# ========================== 评测配置 ==========================
CONFIG = {
    "model_name": "",  # 通过命令行参数传入
    "dataset_dir": "",
    "eval_file": "tst.json",
    "train_file": "trn.json", 
    "train_aug_file": "trn-aug.json",
    "test_filter_file": 'filter_labels_test.txt',
    "train_filter_file": 'filter_labels_train.txt',
    "label_pool_file": "lbl.json",
    "max_length": 512,
    "batch_size": 128,
    "bf16": True,
    "top_k": 100,
    "use_label_augmentation": False,  # 标签增强开关
}


# ========================== 必要的工具类与函数 ==========================
def mean_pooling(model_output, attention_mask):
    """均值池化获取句子嵌入"""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def get_embeddings(model, input_ids, attention_mask, normalize=True):
    """获取文本嵌入，支持Arctic特殊池化"""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    # 修复冗余参数，直接根据CONFIG判断
    if 'arctic' in CONFIG['model_name'].lower():
        embeddings = outputs[0][:, 0]  # Arctic使用[CLS]
    else:
        embeddings = mean_pooling(outputs, attention_mask)
    if normalize:
        return torch.nn.functional.normalize(embeddings, p=2, dim=-1)
    return embeddings


def load_label_texts(label_file, dataset_dir):
    """加载标签文本列表"""
    path = os.path.join(dataset_dir, label_file)
    labels = []
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            labels = [json.loads(line)['title'] for line in f.readlines()]
    return labels


def compute_metrics(retrieved_indices, ground_truth_sets, max_k=100):
    """计算P@k, R@k, F1@k, nDCG@k, MRR@100指标"""
    ks = [1, 3, 5, 10, 20, 30, 50, 100]
    metric_names = []
    for k in ks:
        metric_names.extend([f'P@{k}', f'R@{k}', f'F1@{k}', f'nDCG@{k}'])
    metric_names.append('MRR@100')
    metrics = {k: 0.0 for k in metric_names}
    n_valid = 0
    log_inv = [1.0 / math.log2(i + 2) for i in range(max_k + 2)]

    for ret, gt in zip(retrieved_indices, ground_truth_sets):
        if not gt: 
            continue
        n_valid += 1
        rel = [1 if (idx != -1 and idx in gt) else 0 for idx in ret]
        
        for k in ks:
            hits = sum(rel[:k])
            p_k = hits / k
            r_k = hits / len(gt)
            f1_k = 2 * p_k * r_k / (p_k + r_k) if (p_k + r_k) > 0 else 0.0
            dcg = sum(rel[i] * log_inv[i] for i in range(k))
            idcg = sum(log_inv[i] for i in range(min(len(gt), k)))
            
            metrics[f'P@{k}'] += p_k
            metrics[f'R@{k}'] += r_k
            metrics[f'F1@{k}'] += f1_k
            metrics[f'nDCG@{k}'] += dcg / idcg if idcg > 0 else 0.0
            
        for r in range(min(len(ret), 100)):
            if rel[r] == 1:
                metrics['MRR@100'] += 1.0 / (r + 1)
                break
                
    for k in metrics:
        metrics[k] /= n_valid if n_valid > 0 else 1.0
    return metrics, n_valid


def gpu_chunked_retrieval(q_embs, l_embs, top_k, q_chunk=1024, l_chunk=20000, device='cuda'):
    """
    GPU分块检索，支持大规模标签池
    优化点：直接接收 torch.Tensor，避免 from_numpy 的冗余转换
    """
    if not torch.cuda.is_available(): 
        device = 'cpu'
        
    # 统一确保输入是 float32 的 Tensor，避免类型冲突
    if isinstance(q_embs, np.ndarray):
        q_embs = torch.from_numpy(q_embs)
    if isinstance(l_embs, np.ndarray):
        l_embs = torch.from_numpy(l_embs)
        
    q_embs = q_embs.float()
    l_embs = l_embs.float()
    
    N_q, D = q_embs.shape
    N_l, _ = l_embs.shape
    final_scores = torch.full((N_q, top_k), -1e9, dtype=torch.float32)
    final_indices = torch.full((N_q, top_k), -1, dtype=torch.int64)

    for q_s in tqdm(range(0, N_q, q_chunk), desc="🔍 Query Chunks", leave=False, ncols=80):
        q_e = min(q_s + q_chunk, N_q)
        # 直接使用 tensor 切片送上设备，无需 from_numpy
        Q = q_embs[q_s:q_e].to(device, non_blocking=True)
        bs = Q.shape[0]
        
        best_s = torch.full((bs, top_k), -1e9, device=device, dtype=torch.float32)
        best_i = torch.full((bs, top_k), -1, device=device, dtype=torch.int64)
        
        for l_s in range(0, N_l, l_chunk):
            l_e = min(l_s + l_chunk, N_l)
            L = l_embs[l_s:l_e].to(device, non_blocking=True)
            
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
        
        # 直接保存回 CPU Tensor
        final_scores[q_s:q_e] = best_s.cpu()
        final_indices[q_s:q_e] = best_i.cpu()
        del Q, best_s, best_i

    return final_scores, final_indices


def apply_filter_to_topk(indices, filter_file, dataset_dir):
    """应用过滤规则到Top-K结果"""
    if filter_file is None:
        return indices
    filter_path = os.path.join(dataset_dir, filter_file)
    if not os.path.exists(filter_path): 
        return indices
    
    mapping = np.loadtxt(filter_path, dtype=int, ndmin=2)
    if mapping.size == 0: 
        return indices

    filter_dict = {}
    for q_idx, l_idx in mapping:
        filter_dict.setdefault(q_idx, set()).add(l_idx)

    N_q, K = indices.shape
    # 兼容 Numpy 和 Tensor
    if isinstance(indices, torch.Tensor):
        filtered = torch.full((N_q, K), -1, dtype=indices.dtype)
    else:
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


# ========================== 网络结构 ==========================
class SingleLayerTransformerEncoder(nn.Module):
    """单层Transformer编码器（用于标签增强）"""
    def __init__(self, d_model=512, n_head=1, dim_feedforward=2048, dropout=0.1, norm_first=False):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_head, dim_feedforward, dropout, 
            activation='gelu', norm_first=norm_first, batch_first=False)
    
    def forward(self, x, src_key_padding_mask=None):
        if isinstance(x, list):
            x = torch.stack(x)
        if src_key_padding_mask is not None:
            output = self.encoder_layer(x, src_key_padding_mask=src_key_padding_mask)
            valid_mask = (src_key_padding_mask == 0).float().transpose(0, 1).unsqueeze(-1)
            valid_mask = valid_mask.expand(-1, -1, output.shape[2])
            masked_output = output * valid_mask
            sum_output = torch.sum(masked_output, dim=0)
            valid_counts = torch.clamp(torch.sum(valid_mask, dim=0), min=1.0)
            return sum_output / valid_counts
        else:
            output = self.encoder_layer(x)
            return torch.mean(output, dim=0)


class Augmenter(nn.Module):
    """标签语义增强模块"""
    def __init__(self, model, config, label_embs_numpy=None, n_clusters=None, 
                 use_fv=True, train_encoder=False):
        super().__init__()
        self.encoder = model
        self.use_fv = use_fv
        self.train_encoder = train_encoder
        self.hidden_size = config.hidden_size
        
        if not self.train_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()
        
        if self.use_fv:
            self.n_clusters = n_clusters if n_clusters is not None else 0
            self.register_buffer("label_mapping", torch.empty(0, dtype=torch.long))
            self.snet_weight = nn.Parameter(torch.Tensor(1, self.hidden_size))
            self.combiner = SingleLayerTransformerEncoder(
                d_model=self.hidden_size, n_head=4, dim_feedforward=512, dropout=0.1)

    def encode_text(self, input_ids, attention_mask):
        """编码Query文本"""
        with torch.no_grad():
            return get_embeddings(self.encoder, input_ids, attention_mask, normalize=True)

    def encode_label_raw(self, input_ids, attention_mask):
        """编码Label文本（原始方式，无增强）"""
        with torch.no_grad():
            t_emb = get_embeddings(self.encoder, input_ids, attention_mask, normalize=True)
        return t_emb
    
    @torch.no_grad()
    def encode_label(self, input_ids, attention_mask, lbl_inds):
        """编码Label文本（带语义增强）"""
        with torch.no_grad():
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

    @classmethod
    def load_augmenter(cls, model_name_or_path, device, use_label_augmentation=True):
        """加载增强器（支持开关）"""
        state_path = os.path.join(model_name_or_path, "augmenter_state.bin")
        dtype = torch.bfloat16 if CONFIG["bf16"] and torch.cuda.is_bf16_supported() else torch.float32

        if use_label_augmentation and os.path.exists(state_path):
            augmenter_state = torch.load(state_path, map_location=device, weights_only=False)
            config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
            model = AutoModel.from_pretrained(model_name_or_path, dtype=dtype, trust_remote_code=True)
            
            augmenter = cls(model, config, label_embs_numpy=None, n_clusters=None, 
                           use_fv=augmenter_state.get('use_fv', True),
                           train_encoder=augmenter_state.get('train_encoder', False))
            
            augmenter.n_clusters = augmenter_state['n_clusters']
            augmenter.label_mapping = augmenter_state['label_mapping'].to(device)
            augmenter.snet_weight = nn.Parameter(augmenter_state['snet_weight'].to(device))
            augmenter.combiner.load_state_dict(augmenter_state['combiner'])
            augmenter.to(device)
            augmenter.eval()
            return augmenter
        else:
            config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
            model = AutoModel.from_pretrained(model_name_or_path, dtype=dtype, trust_remote_code=True)
            augmenter = cls(model, config, use_fv=False, train_encoder=False)
            augmenter.to(device)
            augmenter.eval()
            return augmenter


class FocalBCE(nn.Module):
    """Focal Loss for Binary Classification"""
    __slots__ = ('gamma', 'alpha', 'reduction')
    
    def __init__(self, gamma=2.0, alpha=0.25, reduction='sum'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, logits, targets):
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        loss = focal_weight * ce_loss
        return loss.sum() if self.reduction == 'sum' else (loss.mean() if self.reduction == 'mean' else loss)


class ContextualReranker(nn.Module):
    """轻量级重排序模型：融合Query-Context特征"""
    __slots__ = ('proj_input', 'mlp', 'emb_dim')
    
    def __init__(self, emb_dim=768, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.emb_dim = emb_dim
        input_dim = emb_dim * 5  # [q, c, q*c, q-c, c-mean]
        
        self.proj_input = nn.Linear(input_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, q, c):
        B, K, D = c.shape
        q_exp = q.unsqueeze(1)  # [B, 1, D]
        
        prod = q_exp * c
        diff = q_exp - c
        delta = c - c.mean(dim=1, keepdim=True)
        
        x = torch.cat([q_exp.expand(-1, K, -1), c, prod, diff, delta], dim=-1)
        return self.mlp(self.proj_input(x)).squeeze(-1)


# ========================== 数据工具 ==========================
class RerankDataset(Dataset):
    """重排序训练数据集"""
    __slots__ = ('samples', 'query_embs', 'label_embs', 'pool_size')
    
    def __init__(self, samples_path, query_embs_path, label_embs_path, pool_size=100):
        self.pool_size = pool_size
        self.query_embs = torch.load(query_embs_path, map_location='cpu')
        self.label_embs = torch.load(label_embs_path, map_location='cpu')
        
        with open(samples_path, 'r', encoding='utf-8') as f:
            self.samples = json.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        q_emb = self.query_embs[s['query_idx']]
        c_inds = s['candidate_label_indices']
        labels = s['labels']
        
        # 动态填充，避免在 _preprocess 中生成大量冗余 Tensor 浪费内存
        if len(c_inds) < self.pool_size:
            pad_len = self.pool_size - len(c_inds)
            c_inds = c_inds + [-1] * pad_len
            labels = labels + [0.0] * pad_len
        else:
            c_inds, labels = c_inds[:self.pool_size], labels[:self.pool_size]
            
        c_inds_tensor = torch.tensor(c_inds, dtype=torch.long)
        valid_mask = c_inds_tensor >= 0
        
        c_embs = torch.zeros(self.pool_size, self.query_embs.shape[1])
        if valid_mask.any():
            c_embs[valid_mask] = self.label_embs[c_inds_tensor[valid_mask]]
            
        return {
            'query_emb': q_emb,
            'candidate_embs': c_embs,
            'labels': torch.tensor(labels, dtype=torch.float32)
        }


def collate_fn(batch):
    """批量数据聚合"""
    return {
        'query_emb': torch.stack([x['query_emb'] for x in batch]),
        'candidate_embs': torch.stack([x['candidate_embs'] for x in batch]),
        'labels': torch.stack([x['labels'] for x in batch])
    }


# ========================== 核心功能 ==========================
def encode_all_texts(tokenizer, model, texts, batch_size, max_length, device, dtype, 
                     augmenter=None, use_augmentation=False, is_label=False):
    """批量编码文本，支持标签增强开关。优化点：直接返回 Tensor"""
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size),desc='Encoding texts'):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, max_length=max_length, padding=True, 
                       truncation=True, return_tensors='pt').to(device)
        
        if is_label and use_augmentation and augmenter is not None:
            lbl_inds = torch.arange(i, min(i+batch_size, len(texts))).to(device)
            emb = augmenter.encode_label(enc['input_ids'], enc['attention_mask'], lbl_inds)
        elif is_label and augmenter is not None:
            emb = augmenter.encode_label_raw(enc['input_ids'], enc['attention_mask'])
        else:
            emb = augmenter.encode_text(enc['input_ids'], enc['attention_mask']) if augmenter else \
                  get_embeddings(model, enc['input_ids'], enc['attention_mask'], normalize=True)
        
        embeddings.append(emb.cpu())
    
    # 优化：直接返回 float32 的 Tensor，避免后续冗余转换
    return torch.cat(embeddings, dim=0).float()


def generate_rerank_samples(query_embs, label_embs, ground_truth_sets, 
                           topk=100, filter_file=None, dataset_dir=None):
    """生成重排序训练样本。优化点：直接处理 Tensor，避免 Numpy 互转"""
    # 粗排检索 (内部已兼容 Tensor 输入)
    _, coarse_indices = gpu_chunked_retrieval(
        query_embs, label_embs, top_k=topk, q_chunk=256, l_chunk=20000)
    
    # 应用过滤 (内部已兼容 Tensor 输入)
    if filter_file and dataset_dir:
        coarse_indices = apply_filter_to_topk(coarse_indices, filter_file, dataset_dir)
    
    # 转为 numpy 用于快速 Python 遍历生成 JSON 样本
    if isinstance(coarse_indices, torch.Tensor):
        coarse_indices = coarse_indices.numpy()
        
    samples = []
    for i, (indices, gt_set) in enumerate(zip(coarse_indices, ground_truth_sets)):
        valid_indices = [int(idx) for idx in indices if idx != -1]
        labels = [1.0 if idx in gt_set else 0.0 for idx in valid_indices]
        
        samples.append({
            'query_idx': i,
            'candidate_label_indices': valid_indices,
            'labels': labels
        })
    
    return samples


@torch.inference_mode()
def rerank_batch(model, query_embs, candidate_embs, device, dtype, batch_size=64):
    """批量重排序推理。优化点：直接接收 Tensor，避免 from_numpy 转换"""
    model.eval().to(dtype).to(device)
    
    # 统一转为 Tensor
    if isinstance(query_embs, np.ndarray):
        query_embs = torch.from_numpy(query_embs)
    if isinstance(candidate_embs, np.ndarray):
        candidate_embs = torch.from_numpy(candidate_embs)
        
    query_embs = query_embs.to(device, dtype=dtype)
    candidate_embs = candidate_embs.to(device, dtype=dtype)
    
    all_scores = []
    for i in tqdm(range(0, len(query_embs), batch_size),desc="Reranking"):
        q_batch = query_embs[i:i+batch_size]
        c_batch = candidate_embs[i:i+batch_size]
        
        with torch.autocast('cuda', dtype=dtype, enabled=(dtype != torch.float32)):
            scores = model(q_batch, c_batch)
        all_scores.append(scores.cpu())
    
    # 直接返回 Tensor，避免 numpy 转换
    return torch.cat(all_scores, dim=0).float()


def train_reranker(train_samples_path, query_embs_path, label_embs_path, 
                   output_dir, args, dtype, device):
    """训练重排序模型"""
    dataset = RerankDataset(train_samples_path, query_embs_path, label_embs_path, 
                           pool_size=args.candidate_pool)
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                           collate_fn=collate_fn, num_workers=2, pin_memory=True)
    
    model = ContextualReranker(emb_dim=args.emb_dim, hidden_dim=args.hidden_dim).to(dtype).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = FocalBCE(gamma=2.0, alpha=0.25)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    best_loss, patience = float('inf'), 0
    model_dir = os.path.join(output_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in pbar:
            q = batch['query_emb'].to(device, dtype=dtype, non_blocking=True)
            c = batch['candidate_embs'].to(device, dtype=dtype, non_blocking=True)
            y = batch['labels'].to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.autocast('cuda', dtype=dtype, enabled=(dtype != torch.float32)):
                logits = model(q, c)
                loss = criterion(logits.float(), y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'Loss': f'{epoch_loss/(pbar.n+1):.4f}'})
        
        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience = 0
            torch.save(model.state_dict(), os.path.join(model_dir, "reranker-best.pth"))
        else:
            patience += 1
            if patience >= args.patience:
                print(f"⏹️ Early stopping at epoch {epoch+1}")
                break
    
    torch.save(model.state_dict(), os.path.join(model_dir, "reranker-final.pth"))
    print(f"✓ Training completed. Best loss: {best_loss:.4f}")
    return model


@torch.inference_mode()
def evaluate_reranker(model, test_samples_path, query_embs, label_embs, 
                     ground_truth_sets, args, dtype, device, topk=100):
    """评估重排序效果。优化点：向量化提取候选嵌入，避免低效 for 循环"""
    with open(test_samples_path, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    
    # 优化：使用 PyTorch 向量化索引替代 Python 循环拼接 Numpy
    candidate_embs = torch.zeros(len(samples), args.candidate_pool, label_embs.shape[1])
    c_inds_matrix = torch.full((len(samples), args.candidate_pool), -1, dtype=torch.long)
    
    for i, s in enumerate(samples):
        c_inds = s['candidate_label_indices']
        length = min(len(c_inds), args.candidate_pool)
        if length > 0:
            c_inds_tensor = torch.tensor(c_inds[:length], dtype=torch.long)
            valid_mask = c_inds_tensor >= 0
            if valid_mask.any():
                c_inds_matrix[i, :length][valid_mask] = c_inds_tensor[valid_mask]
    
    valid_mask = c_inds_matrix >= 0
    # 利用高级索引一次性填充所有有效的候选嵌入
    if valid_mask.any():
        q_indices, k_indices = torch.where(valid_mask)
        l_indices = c_inds_matrix[valid_mask]
        candidate_embs[q_indices, k_indices] = label_embs[l_indices]
    
    # 重排序推理 (内部已适配 Tensor)
    scores = rerank_batch(model, query_embs, candidate_embs, device, dtype, batch_size=args.batch_size)
    
    # 生成最终排序结果
    final_indices = np.full((len(samples), topk), -1, dtype=np.int64)
    for i, (score_row, sample) in enumerate(zip(scores, samples)):
        c_inds = np.array(sample['candidate_label_indices'])
        valid_mask = c_inds != -1
        if valid_mask.any():
            valid_scores = score_row[valid_mask].numpy()
            valid_inds = c_inds[valid_mask]
            k = min(topk, len(valid_scores))
            topk_pos = np.argsort(-valid_scores)[:k]
            final_indices[i, :k] = valid_inds[topk_pos]
    
    # 计算指标
    metrics, n_valid = compute_metrics(final_indices.tolist(), ground_truth_sets, max_k=topk)
    return metrics


def print_metrics_table(metrics, prefix=""):
    """格式化打印评估指标"""
    ks = [1, 3, 5, 10, 20, 30, 50, 100]
    print(f"\n{'='*70}")
    print(f"{prefix} Evaluation Metrics")
    print(f"{'='*70}")
    print(f"{'K':<5} | {'P@k':<10} | {'R@k':<10} | {'F1@k':<10} | {'nDCG@k':<10}")
    print(f"{'-'*5} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10}")
    for k in ks:
        print(f"{k:<5} | {metrics[f'P@{k}']:<10.4f} | {metrics[f'R@{k}']:<10.4f} | "
              f"{metrics[f'F1@{k}']:<10.4f} | {metrics[f'nDCG@{k}']:<10.4f}")
    print(f"\nMRR@100: {metrics['MRR@100']:.4f}")
    print(f"{'='*70}\n")


# ========================== 主流程 ==========================
def main():
    parser = argparse.ArgumentParser(description="Rerank Training & Evaluation")
    parser.add_argument("--model_path", type=str, required=True, help="Base model path")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Dataset directory")
    parser.add_argument("--output_dir", type=str, default="Results/rerank", help="Output directory")
    parser.add_argument("--use_label_augmentation", action="store_true", help="Enable label augmentation")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--embed_batch_size", type=int, default=128)
    parser.add_argument("--candidate_pool", type=int, default=100)
    parser.add_argument("--retrieval_topk", type=int, default=100)
    parser.add_argument("--rerank_topk", type=int, default=100)
    parser.add_argument("--final_topk", type=int, default=100)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--precision", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--skip_gen", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    # 配置更新
    CONFIG.update({
        "model_name": args.model_path,
        "dataset_dir": args.dataset_dir,
        "max_length": args.max_length,
        "batch_size": args.embed_batch_size,
        "bf16": args.precision == "bfloat16",
        "top_k": args.retrieval_topk,
        "use_label_augmentation": args.use_label_augmentation
    })
    print(CONFIG)

    # 精度设置
    dtype = torch.bfloat16 if args.precision == "bfloat16" and torch.cuda.is_bf16_supported() else \
            torch.float16 if args.precision == "float16" else torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化
    args.emb_dim = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True).hidden_size
    output_dir = os.path.join(args.output_dir, os.path.basename(args.dataset_dir))
    logger = setup_logger(output_dir)
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    logger.info(f"🔧 Config: precision={args.precision}, aug={args.use_label_augmentation}")
    
    # 加载模型与分词器
    logger.info(f"📦 Loading model from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    augmenter = Augmenter.load_augmenter(
        args.model_path, device, use_label_augmentation=args.use_label_augmentation)
    
    # 加载数据
    logger.info("📚 Loading datasets...")
    label_texts = load_label_texts(CONFIG["label_pool_file"], args.dataset_dir)
    
    # 加载评估集
    eval_path = os.path.join(args.dataset_dir, CONFIG["eval_file"])
    eval_examples = []
    with open(eval_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            query = item.get('title', '') + ': ' + item.get('content', '')
            eval_examples.append({
                "query": query[:CONFIG["max_length"]*10],
                "target_ind": item["target_ind"]
            })
    gt_sets = [set(ex["target_ind"]) for ex in eval_examples]
    
    # 编码阶段 (返回的已经是 Tensor)
    logger.info(f"❓ Encoding {len(eval_examples)} queries...")
    query_texts = [ex["query"] for ex in eval_examples]
    query_embs = encode_all_texts(
        tokenizer, augmenter.encoder, query_texts, 
        CONFIG["batch_size"], CONFIG["max_length"], device, dtype,
        augmenter=augmenter, use_augmentation=False, is_label=False)
    
    logger.info(f"🏷️ Encoding {len(label_texts)} labels (aug={args.use_label_augmentation})...")
    label_embs = encode_all_texts(
        tokenizer, augmenter.encoder, label_texts,
        CONFIG["batch_size"], CONFIG["max_length"], device, dtype,
        augmenter=augmenter, use_augmentation=args.use_label_augmentation, is_label=True)
    
    # 数据生成
    data_dir = os.path.join(output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    if not args.skip_gen and not args.eval_only:
        logger.info("🔄 Generating rerank training samples...")
        train_path = os.path.join(args.dataset_dir, CONFIG["train_file"])
        if os.path.exists(train_path):
            train_examples = []
            with open(train_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    query = item.get('title', '') + ': ' + item.get('content', '')
                    train_examples.append({
                        "query": query[:CONFIG["max_length"]*10],
                        "target_ind": item["target_ind"]
                    })
            
            train_aug_path= os.path.join(args.dataset_dir, CONFIG["train_aug_file"])
            if os.path.exists(train_aug_path):
                with open(train_aug_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        item = json.loads(line)
                        query = item.get('title', '') + ': ' + item.get('content', '')
                        train_examples.append({
                            "query": query[:CONFIG["max_length"]*10],
                            "target_ind": item["target_ind"]
                        })
            else:
                raise FileNotFoundError(f"Train aug file not found: {train_aug_path}")
            train_gt = [set(ex["target_ind"]) for ex in train_examples]
            
            # 编码训练集query
            train_query_texts = [ex["query"] for ex in train_examples]
            train_query_embs = encode_all_texts(
                tokenizer, augmenter.encoder, train_query_texts,
                CONFIG["batch_size"], CONFIG["max_length"], device, dtype,
                augmenter=augmenter, use_augmentation=False, is_label=False)
            
            samples = generate_rerank_samples(
                train_query_embs, label_embs, train_gt,
                topk=args.retrieval_topk,
                filter_file=CONFIG["train_filter_file"],
                dataset_dir=args.dataset_dir)
            
            samples_path = os.path.join(data_dir, "rerank_train_samples.json")
            with open(samples_path, 'w', encoding='utf-8') as f:
                json.dump(samples, f, ensure_ascii=False)
            
            # 优化：直接保存 Tensor，无需 from_numpy 转换
            torch.save(train_query_embs, os.path.join(data_dir, "trn_query_embs.pt"))
            torch.save(label_embs, os.path.join(data_dir, "label_embs.pt"))
            logger.info(f"✓ Saved {len(samples)} training samples")
    
    # 训练阶段
    if not args.skip_train and not args.eval_only:
        logger.info("🎯 Training reranker...")
        train_reranker(
            os.path.join(data_dir, "rerank_train_samples.json"),
            os.path.join(data_dir, "trn_query_embs.pt"),
            os.path.join(data_dir, "label_embs.pt"),
            output_dir, args, dtype, device)
    
    # 评估阶段
    logger.info("📊 Evaluating...")
    
    # ============ 新增：未重排（粗排）结果评估 ============
    logger.info("📊 [1/2] Evaluating Coarse Retrieval (No Rerank)...")
    _, coarse_indices_eval = gpu_chunked_retrieval(
        query_embs, label_embs, top_k=args.final_topk, q_chunk=256, l_chunk=20000)
    
    if CONFIG["test_filter_file"]:
        coarse_indices_eval = apply_filter_to_topk(
            coarse_indices_eval, CONFIG["test_filter_file"], args.dataset_dir)
        
    # 兼容 Tensor 和 numpy
    if isinstance(coarse_indices_eval, torch.Tensor):
        coarse_indices_eval = coarse_indices_eval.numpy()
        
    coarse_metrics, _ = compute_metrics(coarse_indices_eval.tolist(), gt_sets, max_k=args.final_topk)
    aug_tag = "[Augmented] " if args.use_label_augmentation else "[Raw] "
    print_metrics_table(coarse_metrics, prefix=f"{aug_tag}Coarse (No Rerank)")
    # ============ 新增结束 ============


    model_path = os.path.join(output_dir, "model", "reranker-best.pth")
    if os.path.exists(model_path):
        reranker = ContextualReranker(emb_dim=args.emb_dim, hidden_dim=args.hidden_dim)
        reranker.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False))
        
        # 保存测试样本用于评估
        test_samples = generate_rerank_samples(
            query_embs, label_embs, gt_sets,
            topk=args.retrieval_topk,
            filter_file=CONFIG["test_filter_file"],
            dataset_dir=args.dataset_dir)
        test_samples_path = os.path.join(data_dir, "rerank_test_samples.json")
        with open(test_samples_path, 'w', encoding='utf-8') as f:
            json.dump(test_samples, f, ensure_ascii=False)
        
        # 传入 Tensor 给评估函数
        logger.info("📊 [2/2] Evaluating Reranker...")
        metrics = evaluate_reranker(
            reranker, test_samples_path, query_embs, label_embs,
            gt_sets, args, dtype, device, topk=args.final_topk)
        
        print_metrics_table(metrics, prefix=f"{aug_tag}Rerank")
        
        # 保存结果
        eval_dir = os.path.join(output_dir, "eval")
        os.makedirs(eval_dir, exist_ok=True)
        with open(os.path.join(eval_dir, "metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=2)
    else:
        logger.info("⚠️ No trained reranker found, skipping rerank evaluation")
    
    logger.info(f"✅ Pipeline completed. Outputs: {output_dir}")



if __name__ == "__main__":
    main()
