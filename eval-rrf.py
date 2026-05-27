#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
import os, sys, json, argparse, math, logging, time
from datetime import datetime
from collections import defaultdict
from itertools import combinations  # ← 新增：用于生成两两组合
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel, AutoConfig
from tqdm import tqdm
import numpy as np

# ========================== 日志配置 ==========================
def setup_logger(output_dir):
    """初始化日志记录器"""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logger = logging.getLogger("MultiStrategy_Evaluator")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


# ========================== 评测配置 ==========================
CONFIG = {
    "model_name": "",
    "dataset_dir": "",
    "eval_file": "tst.json",
    "test_filter_file": 'filter_labels_test.txt',
    "label_pool_file": "lbl.json",
    "max_length": 512,
    "batch_size": 128,
    "bf16": True,
    "top_k": 100,
}


# ========================== 工具函数 ==========================
def mean_pooling(model_output, attention_mask):
    """均值池化获取句子嵌入"""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def get_embeddings(model, input_ids, attention_mask, normalize=True, use_arctic_pooling=False):
    """获取文本嵌入，支持Arctic特殊池化"""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    if use_arctic_pooling or 'arctic' in CONFIG["model_name"].lower():
        embeddings = outputs[0][:, 0]
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


def compute_metrics(retrieved_indices, ground_truth_sets, ks=None):
    """计算P@k, R@k, F1@k指标"""
    if ks is None:
        ks = [1, 5, 10, 20, 30, 50, 100]
    
    metrics = {k: {'P': 0.0, 'R': 0.0, 'F1': 0.0} for k in ks}
    n_valid = 0
    
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
            
            metrics[k]['P'] += p_k
            metrics[k]['R'] += r_k
            metrics[k]['F1'] += f1_k
    
    for k in ks:
        for m in ['P', 'R', 'F1']:
            metrics[k][m] /= n_valid if n_valid > 0 else 1.0
    
    return metrics, n_valid


def gpu_chunked_retrieval(q_embs, l_embs, top_k, q_chunk=1024, l_chunk=20000, device='cuda'):
    """GPU分块检索，支持大规模标签池"""
    if not torch.cuda.is_available():
        device = 'cpu'
    
    if isinstance(q_embs, np.ndarray):
        q_embs = torch.from_numpy(q_embs)
    if isinstance(l_embs, np.ndarray):
        l_embs = torch.from_numpy(l_embs)
    
    q_embs = q_embs.float().to(device)
    l_embs = l_embs.float().to(device)
    
    N_q, D = q_embs.shape
    N_l, _ = l_embs.shape
    final_scores = torch.full((N_q, top_k), -1e9, dtype=torch.float32, device='cpu')
    final_indices = torch.full((N_q, top_k), -1, dtype=torch.int64, device='cpu')
    
    for q_s in tqdm(range(0, N_q, q_chunk), desc="🔍 Retrieval", leave=False):
        q_e = min(q_s + q_chunk, N_q)
        Q = q_embs[q_s:q_e]
        bs = Q.shape[0]
        
        best_s = torch.full((bs, top_k), -1e9, device=device, dtype=torch.float32)
        best_i = torch.full((bs, top_k), -1, device=device, dtype=torch.int64)
        
        for l_s in range(0, N_l, l_chunk):
            l_e = min(l_s + l_chunk, N_l)
            L = l_embs[l_s:l_e]
            
            with torch.no_grad():
                sim = torch.matmul(Q, L.t())
            
            k_l = min(top_k, l_e - l_s)
            chunk_s, chunk_i = torch.topk(sim, k=k_l, dim=1, sorted=False)
            chunk_i = chunk_i + l_s
            
            merged_s = torch.cat([best_s, chunk_s], dim=1)
            merged_i = torch.cat([best_i, chunk_i], dim=1)
            best_s, idx_merge = torch.topk(merged_s, k=top_k, dim=1, sorted=True)
            best_i = torch.gather(merged_i, 1, idx_merge)
        
        final_scores[q_s:q_e] = best_s.cpu()
        final_indices[q_s:q_e] = best_i.cpu()
    
    return final_scores.numpy(), final_indices.numpy()


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


def rrf_fusion(results_list, ks, rrf_k=60):
    """
    Reciprocal Rank Fusion (RRF) 融合多个检索结果
    Args:
        results_list: List of (scores, indices) tuples
        ks: 融合后返回的top-k数量
        rrf_k: RRF常数，通常为60
    Returns:
        fused_indices: (N_q, ks) 融合后的索引
    """
    N_q = len(results_list[0][1])
    fused_scores = defaultdict(lambda: defaultdict(float))
    
    for scores, indices in results_list:
        for i in range(N_q):
            for rank, idx in enumerate(indices[i]):
                if idx != -1:
                    fused_scores[i][idx] += 1.0 / (rrf_k + rank + 1)
    
    fused_indices = np.full((N_q, ks), -1, dtype=np.int64)
    for i in range(N_q):
        if fused_scores[i]:
            sorted_items = sorted(fused_scores[i].items(), key=lambda x: -x[1])
            for j, (idx, _) in enumerate(sorted_items[:ks]):
                fused_indices[i, j] = idx
    
    return fused_indices


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
    def __init__(self, model, config, use_fv=True, train_encoder=False):
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
            self.n_clusters = 0
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
            return get_embeddings(self.encoder, input_ids, attention_mask, normalize=True)
    
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
    def load_augmenter(cls, model_name_or_path, device):
        """加载增强器"""
        state_path = os.path.join(model_name_or_path, "augmenter_state.bin")
        dtype = torch.bfloat16 if CONFIG["bf16"] and torch.cuda.is_bf16_supported() else torch.float32
        
        if os.path.exists(state_path):
            augmenter_state = torch.load(state_path, map_location=device, weights_only=False)
            config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
            model = AutoModel.from_pretrained(model_name_or_path, dtype=dtype, trust_remote_code=True)
            
            augmenter = cls(model, config, 
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


class ContextualReranker(nn.Module):
    """轻量级重排序模型"""
    def __init__(self, emb_dim=768, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.emb_dim = emb_dim
        input_dim = emb_dim * 5
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
        q_exp = q.unsqueeze(1)
        prod = q_exp * c
        diff = q_exp - c
        delta = c - c.mean(dim=1, keepdim=True)
        x = torch.cat([q_exp.expand(-1, K, -1), c, prod, diff, delta], dim=-1)
        return self.mlp(self.proj_input(x)).squeeze(-1)


# ========================== 编码函数 ==========================
def encode_texts_batch(tokenizer, model, texts, batch_size, max_length, device, dtype, 
                       augmenter=None, use_augmentation=False, is_label=False, desc="Encoding"):
    """批量编码文本"""
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc=desc, leave=False):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, max_length=max_length, padding=True, 
                       truncation=True, return_tensors='pt').to(device)
       
        if is_label and use_augmentation and augmenter is not None:
            lbl_inds = torch.arange(i, min(i+batch_size, len(texts))).to(device)
            emb = augmenter.encode_label(enc['input_ids'], enc['attention_mask'], lbl_inds)
        elif is_label and augmenter is not None:
            emb = augmenter.encode_label_raw(enc['input_ids'], enc['attention_mask'])
        else:
            emb = get_embeddings(model, enc['input_ids'], enc['attention_mask'], normalize=True)
        
        embeddings.append(emb.cpu())
    
    return torch.cat(embeddings, dim=0).float()


# ========================== 重排序推理 ==========================
@torch.inference_mode()
def rerank_batch(model, query_embs, candidate_embs, device, dtype, batch_size=64):
    """批量重排序推理"""
    model.eval().to(dtype).to(device)
    if isinstance(query_embs, np.ndarray):
        query_embs = torch.from_numpy(query_embs)
    if isinstance(candidate_embs, np.ndarray):
        candidate_embs = torch.from_numpy(candidate_embs)
    
    query_embs = query_embs.to(device, dtype=dtype)
    candidate_embs = candidate_embs.to(device, dtype=dtype)
    
    all_scores = []
    for i in range(0, len(query_embs), batch_size):
        q_batch = query_embs[i:i+batch_size]
        c_batch = candidate_embs[i:i+batch_size]
        with torch.autocast('cuda', dtype=dtype, enabled=(dtype != torch.float32)):
            scores = model(q_batch, c_batch)
        all_scores.append(scores.cpu())
    
    return torch.cat(all_scores, dim=0).float().numpy()


# ========================== 评估函数 ==========================
def evaluate_strategy(query_embs, label_embs, gt_sets, strategy_name, 
                     tokenizer, model, augmenter, device, dtype, args, ks, filter_file=None):
    """评估单种策略"""
    logger = logging.getLogger("MultiStrategy_Evaluator")
    
    logger.info(f"🔍 Evaluating [{strategy_name}]...")
    
    # 1. 粗排检索
    scores, indices = gpu_chunked_retrieval(
        query_embs, label_embs, top_k=args.retrieval_topk, 
        q_chunk=args.batch_size, l_chunk=20000, device=device)
    
    if filter_file:
        indices = apply_filter_to_topk(indices, filter_file, args.dataset_dir)
    
    # 2. 如需重排序
    if "rerank" in strategy_name.lower() and args.reranker_path and os.path.exists(args.reranker_path):
        logger.info(f"   → Applying reranker...")
        reranker = ContextualReranker(emb_dim=args.emb_dim, hidden_dim=args.hidden_dim)
        reranker.load_state_dict(torch.load(args.reranker_path, map_location='cpu', weights_only=False))
        
        # 准备候选嵌入
        candidate_embs = np.zeros((len(query_embs), args.candidate_pool, label_embs.shape[1]))
        for i, idx_row in enumerate(indices):
            valid = [j for j, idx in enumerate(idx_row) if idx != -1]
            if valid:
                candidate_embs[i, :len(valid)] = label_embs[idx_row[valid]]
        
        # 重排序
        rerank_scores = rerank_batch(reranker, query_embs, candidate_embs, device, dtype)
        
        # 生成新排序
        final_indices = np.full((len(query_embs), args.final_topk), -1, dtype=np.int64)
        for i, (score_row, idx_row) in enumerate(zip(rerank_scores, indices)):
            valid_mask = idx_row != -1
            if valid_mask.any():
                valid_scores = score_row[valid_mask]
                valid_inds = idx_row[valid_mask]
                k = min(args.final_topk, len(valid_scores))
                topk_pos = np.argsort(-valid_scores)[:k]
                final_indices[i, :k] = valid_inds[topk_pos]
        indices = final_indices
    
    # 3. 计算指标
    metrics, n_valid = compute_metrics(indices.tolist(), gt_sets, ks=ks)
    
    logger.info(f"   ✓ [{strategy_name}] Valid queries: {n_valid}")
    return indices, metrics


def print_results_table(all_results, ks=None):
    """打印对比结果表格（支持两两融合）"""
    if ks is None:
        ks = [1, 5, 10, 20, 30, 50, 100]
    
    logger = logging.getLogger("MultiStrategy_Evaluator")
    
    # 分类策略：基础策略 / 两两融合 / 全融合
    # 🔥 关键修改：将 "Augmented+Rerank" 加入基础策略列表
    base_strategies = [k for k in all_results if k in ["Raw_NoAug", "Raw+Rerank", "Augmented"]]
    pair_strategies = [k for k in all_results if k.startswith("RRF_") and k.count("+") == 1]
    full_strategy = [k for k in all_results if k == "RRF_Fusion"]
    
    logger.info(f"\n{'='*120}")
    logger.info(f"{'📊 Evaluation Results Comparison':^120}")
    logger.info(f"{'='*120}")
    
    # 打印基础策略
    if base_strategies:
        logger.info(f"\n🔹 Base Strategies:")
        header = f"{'K':<4} | "
        for strat in base_strategies:
            header += f"{strat:^32} | "
        logger.info(header)
        logger.info(f"{'-'*4} | " + "-"*36*len(base_strategies))
        
        for k in ks:
            row = f"{k:<4} | "
            for strat in base_strategies:
                metrics = all_results[strat]['metrics']
                p, r, f1 = metrics[k]['P'], metrics[k]['R'], metrics[k]['F1']
                row += f"P@{k}={p:.4f} R@{k}={r:.4f} F1@{k}={f1:.4f} | "
            logger.info(row)
    
    # 打印两两融合策略
    if pair_strategies:
        logger.info(f"\n🔹 Pairwise RRF Fusion:")
        header = f"{'K':<4} | "
        for strat in pair_strategies:
            header += f"{strat:^32} | "
        logger.info(header)
        logger.info(f"{'-'*4} | " + "-"*36*len(pair_strategies))
        
        for k in ks:
            row = f"{k:<4} | "
            for strat in pair_strategies:
                metrics = all_results[strat]['metrics']
                p, r, f1 = metrics[k]['P'], metrics[k]['R'], metrics[k]['F1']
                row += f"P@{k}={p:.4f} R@{k}={r:.4f} F1@{k}={f1:.4f} | "
            logger.info(row)
    
    # 打印全融合策略
    if full_strategy:
        logger.info(f"\n🔹 Full RRF Fusion:")
        header = f"{'K':<4} | "
        for strat in full_strategy:
            header += f"{strat:^32} | "
        logger.info(header)
        logger.info(f"{'-'*4} | " + "-"*36*len(full_strategy))
        
        for k in ks:
            row = f"{k:<4} | "
            for strat in full_strategy:
                metrics = all_results[strat]['metrics']
                p, r, f1 = metrics[k]['P'], metrics[k]['R'], metrics[k]['F1']
                row += f"P@{k}={p:.4f} R@{k}={r:.4f} F1@{k}={f1:.4f} | "
            logger.info(row)
    
    logger.info(f"\n{'='*120}\n")


# ========================== 主流程 ==========================
def main():
    parser = argparse.ArgumentParser(description="Multi-Strategy Rerank Evaluation with Pairwise Fusion")
    parser.add_argument("--model_path", type=str, required=True, help="Base model path")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Dataset directory")
    parser.add_argument("--output_dir", type=str, default="Results/eval_multi", help="Output directory")
    parser.add_argument("--reranker_path", type=str, default=None, help="Trained reranker path (optional)")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--retrieval_topk", type=int, default=100)
    parser.add_argument("--candidate_pool", type=int, default=100)
    parser.add_argument("--final_topk", type=int, default=100)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--precision", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--rrf_k", type=float, default=60.0, help="RRF constant")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    # 配置更新
    CONFIG.update({
        "model_name": args.model_path,
        "dataset_dir": args.dataset_dir,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "bf16": args.precision == "bfloat16",
        "top_k": args.retrieval_topk,
    })
    
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
    
    logger.info(f"🔧 Config: precision={args.precision}, device={device}")
    
    # 加载模型与分词器
    logger.info(f"📦 Loading model from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    augmenter = Augmenter.load_augmenter(args.model_path, device)
    
    # 加载数据
    logger.info("📚 Loading datasets...")
    label_texts = load_label_texts(CONFIG["label_pool_file"], args.dataset_dir)
    
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
    query_texts = [ex["query"] for ex in eval_examples]
    
    # 编码阶段
    logger.info(f"❓ Encoding {len(query_texts)} queries...")
    query_embs = encode_texts_batch(
        tokenizer, augmenter.encoder, query_texts,
        CONFIG["batch_size"], CONFIG["max_length"], device, dtype,
        augmenter=augmenter, use_augmentation=False, is_label=False, desc="Encoding queries")
    
    # 编码标签（原始方式，用于策略1/2/4）
    logger.info(f"🏷️ Encoding {len(label_texts)} labels (Raw)...")
    label_embs_raw = encode_texts_batch(
        tokenizer, augmenter.encoder, label_texts,
        CONFIG["batch_size"], CONFIG["max_length"], device, dtype,
        augmenter=augmenter, use_augmentation=False, is_label=True, desc="Encoding labels (Raw)")
    
    # 编码标签（增强方式，用于策略3/4）
    logger.info(f"🚀 Encoding {len(label_texts)} labels (Augmented)...")
    label_embs_aug = encode_texts_batch(
        tokenizer, augmenter.encoder, label_texts,
        CONFIG["batch_size"], CONFIG["max_length"], device, dtype,
        augmenter=augmenter, use_augmentation=True, is_label=True, desc="Encoding labels (Aug)")
    
    # 评估基础策略
    all_results = {}
    ks = [1, 3, 5, 10, 20, 30, 50, 100]
    
    # 策略1: 无标签增强（原始基线）
    indices_raw, metrics_raw = evaluate_strategy(
        query_embs, label_embs_raw.numpy(), gt_sets, "Raw_NoAug",
        tokenizer, augmenter.encoder, augmenter, device, dtype, args,ks=ks,
        filter_file=CONFIG["test_filter_file"])
    all_results["Raw_NoAug"] = {"indices": indices_raw, "metrics": metrics_raw}
    
    # 策略2: 无标签增强 + 重排序
    if args.reranker_path and os.path.exists(args.reranker_path):
        indices_rerank, metrics_rerank = evaluate_strategy(
            query_embs, label_embs_raw.numpy(), gt_sets, "Raw+Rerank",
            tokenizer, augmenter.encoder, augmenter, device, dtype, args, ks=ks,
            filter_file=CONFIG["test_filter_file"])
        all_results["Raw+Rerank"] = {"indices": indices_rerank, "metrics": metrics_rerank}
    else:
        logger.warning("⚠️ Reranker path not provided or not found, skipping Raw+Rerank strategy")
    
    # 策略3: 有标签增强（无重排序）
    indices_aug, metrics_aug = evaluate_strategy(
        query_embs, label_embs_aug.numpy(), gt_sets, "Augmented",
        tokenizer, augmenter.encoder, augmenter, device, dtype, args, ks=ks,
        filter_file=CONFIG["test_filter_file"])
    all_results["Augmented"] = {"indices": indices_aug, "metrics": metrics_aug}
    
    # ========== 新增：策略4 - 有标签增强 + 重排序 ==========
    if args.reranker_path and os.path.exists(args.reranker_path):
        logger.info(f"🔍 Evaluating [Augmented+Rerank]...")
        
        # 1. 使用增强标签嵌入进行粗排检索
        scores_aug, indices_aug_coarse = gpu_chunked_retrieval(
            query_embs, label_embs_aug.numpy(), top_k=args.retrieval_topk, 
            q_chunk=args.batch_size, l_chunk=20000, device=device)
        
        if CONFIG["test_filter_file"]:
            indices_aug_coarse = apply_filter_to_topk(indices_aug_coarse, CONFIG["test_filter_file"], args.dataset_dir)
        
        # 2. 加载重排序模型
        reranker = ContextualReranker(emb_dim=args.emb_dim, hidden_dim=args.hidden_dim)
        reranker.load_state_dict(torch.load(args.reranker_path, map_location='cpu', weights_only=False))
        
        # 3. 准备候选嵌入（使用增强后的标签嵌入）
        candidate_embs = np.zeros((len(query_embs), args.candidate_pool, label_embs_aug.shape[1]))
        for i, idx_row in enumerate(indices_aug_coarse):
            valid = [j for j, idx in enumerate(idx_row) if idx != -1]
            if valid:
                candidate_embs[i, :len(valid)] = label_embs_aug.numpy()[idx_row[valid]]
        
        # 4. 重排序推理
        rerank_scores = rerank_batch(reranker, query_embs, candidate_embs, device, dtype)
        
        # 5. 生成最终排序
        final_indices = np.full((len(query_embs), args.final_topk), -1, dtype=np.int64)
        for i, (score_row, idx_row) in enumerate(zip(rerank_scores, indices_aug_coarse)):
            valid_mask = idx_row != -1
            if valid_mask.any():
                valid_scores = score_row[valid_mask]
                valid_inds = idx_row[valid_mask]
                k = min(args.final_topk, len(valid_scores))
                topk_pos = np.argsort(-valid_scores)[:k]
                final_indices[i, :k] = valid_inds[topk_pos]
        
        # 6. 计算指标
        metrics_aug_rerank, n_valid = compute_metrics(final_indices.tolist(), gt_sets, ks=ks)
        all_results["Augmented+Rerank"] = {"indices": final_indices, "metrics": metrics_aug_rerank}
        logger.info(f"   ✓ [Augmented+Rerank] Valid queries: {n_valid}")
    else:
        logger.warning("⚠️ Reranker path not provided or not found, skipping Augmented+Rerank strategy")
    # ========== 新增结束 ==========
    
    # ========== 两两融合评估（自动包含新策略）==========
    logger.info(f"🔗 Applying Pairwise RRF Fusion (k={args.rrf_k})...")
    
    # 准备所有可用策略的融合输入
    available_strategies = ["Raw_NoAug", "Raw+Rerank", "Augmented", "Augmented+Rerank"]
    strategy_data = {}
    for name in available_strategies:
        if name in all_results:
            indices = all_results[name]["indices"]
            scores = np.zeros_like(indices, dtype=np.float32)
            for i in range(len(indices)):
                for rank, idx in enumerate(indices[i]):
                    if idx != -1:
                        scores[i, rank] = 1.0 / (args.rrf_k + rank + 1)
            strategy_data[name] = (scores, indices)
    
    # 策略5: 全融合 (融合所有可用策略)
    if len(strategy_data) >= 2:
        logger.info(f"🔗 Applying Full RRF Fusion (k={args.rrf_k})...")
        fusion_inputs = [strategy_data[k] for k in strategy_data]
        indices_fusion = rrf_fusion(fusion_inputs, ks=args.final_topk, rrf_k=args.rrf_k)
        
        if CONFIG["test_filter_file"]:
            indices_fusion = apply_filter_to_topk(indices_fusion, CONFIG["test_filter_file"], args.dataset_dir)
        
        metrics_fusion, _ = compute_metrics(indices_fusion.tolist(), gt_sets, ks=ks)
        all_results["RRF_Fusion"] = {"indices": indices_fusion, "metrics": metrics_fusion}
        logger.info(f"   ✓ [RRF_Fusion] Fused {len(fusion_inputs)} strategies")
    else:
        logger.warning("⚠️ Need at least 2 strategies for full RRF fusion")
    
    # ========== 融合结束 ==========
    
    # 打印结果表格（自动包含新策略）
    print_results_table(all_results, ks=ks)
    
    # 保存结果
    eval_dir = os.path.join(output_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    
    results_to_save = {}
    for name, data in all_results.items():
        results_to_save[name] = {
            "metrics": {k: {"P": float(v['P']), "R": float(v['R']), "F1": float(v['F1'])} 
                       for k, v in data["metrics"].items()},
            "indices": data["indices"].tolist() if isinstance(data["indices"], np.ndarray) else data["indices"]
        }
    
    with open(os.path.join(eval_dir, "results.json"), 'w', encoding='utf-8') as f:
        json.dump(results_to_save, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ Evaluation completed. Results saved to: {eval_dir}")
    
    # 输出最佳策略
    best_strategy = None
    best_f1 = -1
    for name, data in all_results.items():
        avg_f1 = sum(data["metrics"][k]["F1"] for k in ks) / len(ks)
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_strategy = name
    
    if best_strategy:
        logger.info(f"🏆 Best Strategy: {best_strategy} (Avg F1@{ks}: {best_f1:.4f})")


if __name__ == "__main__":
    main()
