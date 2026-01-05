# here put the import lib
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm
from models.BaseModel import BaseSeqModel
from models.SASRec import SASRecBackbone
from models.utils import Contrastive_Loss2, cal_bpr_loss


class GraphLearner(nn.Module):
    """
    Adaptive Graph Structure Learner (AGSL).
    Learns to assign probabilities to candidate edges provided by LLM.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # Logits for edge existence
        )

    def forward(self, node_emb_u, node_emb_i):
        """
        node_emb_u: [E, D] embedding of user nodes
        node_emb_i: [E, D] embedding of item nodes
        Returns: logits [E, 1]
        """
        # Concatenate features to learn relationship
        edge_feat = torch.cat([node_emb_u, node_emb_i], dim=-1)
        logits = self.mlp(edge_feat)
        return logits


class LLM4CDSR_base(BaseSeqModel):

    def __init__(self, user_num, item_num_dict, device, args) -> None:
        
        self.item_numA, self.item_numB = item_num_dict["0"], item_num_dict["1"]
        item_num =  self.item_numA + self.item_numB

        super().__init__(user_num, item_num, device, args)

        self.global_emb = args.global_emb

        llm_emb_A = pickle.load(open("./data/{}/handled/{}_A_pca128.pkl".format(args.dataset, args.llm_emb_file), "rb"))
        llm_emb_B = pickle.load(open("./data/{}/handled/{}_B_pca128.pkl".format(args.dataset, args.llm_emb_file), "rb"))
        llm_emb_all = pickle.load(open("./data/{}/handled/{}_all.pkl".format(args.dataset, args.llm_emb_file), "rb"))
        
        llm_item_emb = np.concatenate([
            np.zeros((1, llm_emb_all.shape[1])),
            llm_emb_all
        ])
        if args.global_emb:
            self.item_emb_llm = nn.Embedding.from_pretrained(torch.Tensor(llm_item_emb), padding_idx=0)
        else:
            self.item_emb_llm = nn.Embedding(self.item_numA+self.item_numB+1, args.hidden_size, padding_idx=0)
        
        if args.freeze_emb:
            self.item_emb_llm.weight.requires_grad = False
        else:
            self.item_emb_llm.weight.requires_grad = True
            
        self.adapter = nn.Sequential(
            nn.Linear(llm_item_emb.shape[1], int(llm_item_emb.shape[1] / 2)),
            nn.Linear(int(llm_item_emb.shape[1] / 2), args.hidden_size)
        )

        llm_item_device_str = getattr(args, 'llm_item_device', '')
        if isinstance(llm_item_device_str, str) and len(llm_item_device_str) > 0:
            try:
                self.llm_item_device = torch.device(llm_item_device_str)
            except Exception:
                self.llm_item_device = self.dev
        else:
            self.llm_item_device = self.dev
            
        self.item_emb_llm = self.item_emb_llm.to(self.llm_item_device)
        self.adapter = self.adapter.to(self.llm_item_device)

        self.pos_emb = nn.Embedding(args.max_len+1, args.hidden_size)
        self.emb_dropout = nn.Dropout(p=args.dropout_rate)
        self.backbone = SASRecBackbone(device, args)

        # for domain A
        if args.local_emb:
            llm_embA = np.concatenate([np.zeros((1, llm_emb_A.shape[1])), llm_emb_A])
            self.item_embA = nn.Embedding.from_pretrained(torch.Tensor(llm_embA), padding_idx=0)
        else:
            self.item_embA = nn.Embedding(self.item_numA+1, args.hidden_size, padding_idx=0)
        self.pos_embA = nn.Embedding(args.max_len+1, args.hidden_size)
        self.emb_dropoutA = nn.Dropout(p=args.dropout_rate)
        self.backboneA = SASRecBackbone(device, args)

        # for domain B
        if args.local_emb:
            llm_embB = np.concatenate([np.zeros((1, llm_emb_B.shape[1])), llm_emb_B])
            self.item_embB = nn.Embedding.from_pretrained(torch.Tensor(llm_embB), padding_idx=0)
        else:
            self.item_embB = nn.Embedding(self.item_numB+1, args.hidden_size, padding_idx=0)
        self.pos_embB = nn.Embedding(args.max_len+1, args.hidden_size)
        self.emb_dropoutB = nn.Dropout(p=args.dropout_rate)
        self.backboneB = SASRecBackbone(device, args)

        self.loss_func = nn.BCEWithLogitsLoss(reduction="none")
        
        # --- Cross-Domain LightGCN setup ---
        self.use_gnn = getattr(args, 'use_gnn', False)
        self.gnn_layers = int(getattr(args, 'layer_num', 2))
        
        if self.use_gnn:
            self.user_gnn = nn.Embedding(self.user_num+1, args.hidden_size, padding_idx=0)
            self.item_gnn = nn.Embedding(self.item_num+1, args.hidden_size, padding_idx=0)
            nn.init.xavier_uniform_(self.user_gnn.weight)
            nn.init.xavier_uniform_(self.item_gnn.weight)
            
            self.edge_u = None
            self.edge_i = None
            self.user_deg = None
            self.item_deg = None
            
            # [AGSL] Placeholders for augmented edges
            self.aug_edge_u = None
            self.aug_edge_i = None
            self.aug_prior_scores = None

            self._gnn_item_all = None

        if args.global_emb:
            self.filter_init_modules.append("item_emb_llm")
        if args.local_emb:
            self.filter_init_modules.append("item_embA")
            self.filter_init_modules.append("item_embB")
        self._init_weights()


    def _build_cross_domain_graph(self, args):
        print("Building Cross-Domain Graph...")
        inter_seq, domain_seq = pickle.load(open('./data/{}/handled/{}.pkl'.format(args.dataset, args.inter_file), 'rb'))
        u_list, i_list = [], []
        for u, items in inter_seq.items():
            try:
                uid = int(u)
            except:
                uid = u
            if uid > self.user_num: continue

            dom = domain_seq[u]
            for idx, itm in enumerate(items):
                if itm == 0:
                    continue
                d = int(dom[idx])
                if d == 0:
                    ab_id = int(itm)
                else:
                    ab_id = int(itm) + self.item_numA
                
                u_list.append(uid)
                i_list.append(ab_id)
        
        self.edge_u = torch.tensor(u_list, dtype=torch.long, device=torch.device('cpu'))
        self.edge_i = torch.tensor(i_list, dtype=torch.long, device=torch.device('cpu'))
        
        # Keep fixed edges separately from learned edges
        self.user_deg = torch.bincount(self.edge_u, minlength=self.user_num+1).float().clamp(min=1.0).unsqueeze(-1)
        self.item_deg = torch.bincount(self.edge_i, minlength=self.item_num+1).float().clamp(min=1.0).unsqueeze(-1)
        print("Graph built. Edges:", len(u_list))

    def _get_struct_item_emb(self, item_ids, aug_weights=None, domain="AB"):
        """
        Modified to support weighted propagation from augmented edges.
        aug_weights: [E_aug, 1] soft weights for augmented edges.
        """
        if not self.use_gnn:
            return torch.zeros((*item_ids.shape, self.adapter[-1].out_features), device=self.dev)
        
        if self.edge_u.device.type == 'cpu':
             self.edge_u = self.edge_u.to(self.dev)
             self.edge_i = self.edge_i.to(self.dev)
             self.user_deg = self.user_deg.to(self.dev)
             self.item_deg = self.item_deg.to(self.dev)

        U = self.user_gnn.weight
        I = self.item_gnn.weight
        accI = I
        
        # Propagate
        for _ in range(self.gnn_layers):
             # 1. Standard edges (Fixed, Weight=1.0)
             user_sum = torch.zeros_like(U)
             user_sum.index_add_(0, self.edge_u, I[self.edge_i])
             
             # 2. Augmented edges (Learned, Weight=aug_weights)
             if aug_weights is not None and self.aug_edge_u is not None:
                 # Message = I[aug_i] * weight
                 msgs = I[self.aug_edge_i] * aug_weights
                 user_sum.index_add_(0, self.aug_edge_u, msgs)

             U_next = user_sum / self.user_deg # Note: deg is approx fixed or needs update? Fixed for stability.
             
             # Item <- Users
             item_sum = torch.zeros_like(I)
             item_sum.index_add_(0, self.edge_i, U[self.edge_u])
             
             # Aug edges for items
             if aug_weights is not None and self.aug_edge_u is not None:
                 msgs_u = U[self.aug_edge_u] * aug_weights
                 item_sum.index_add_(0, self.aug_edge_i, msgs_u)
                 
             I_next = item_sum / self.item_deg
             
             U = U_next
             I = I_next
             accI = accI + I
             
        gnn_item_all = accI / (self.gnn_layers + 1)
        
        # Mapping IDs
        if domain == "A":
            ab_ids = item_ids
        elif domain == "B":
            ab_ids = item_ids + self.item_numA
        else:
            ab_ids = item_ids
            
        return gnn_item_all[ab_ids]

    def _get_embedding(self, log_seqs, aug_weights=None, domain="A"):
        # 1. Get Semantic Embedding
        if domain == "A":
            item_seq_emb = self.item_embA(log_seqs)
        elif domain == "B":
            item_seq_emb = self.item_embB(log_seqs)
        elif domain == "AB":
            ids_on = log_seqs.to(self.llm_item_device)
            item_seq_emb = self.item_emb_llm(ids_on)
            item_seq_emb = self.adapter(item_seq_emb)
            item_seq_emb = item_seq_emb.to(self.dev)
        else:
            raise ValueError

        # 2. Get Structure Embedding & Fuse
        if self.use_gnn:
            # Pass aug_weights to structure learning
            struct_emb = self._get_struct_item_emb(log_seqs, aug_weights=aug_weights, domain=domain)
            
            combined = torch.cat([item_seq_emb, struct_emb], dim=-1)
            alpha = self.fusion_gate(combined) 
            item_seq_emb = alpha * item_seq_emb + (1 - alpha) * struct_emb
            
        return item_seq_emb
    

    def log2feats(self, log_seqs, positions, aug_weights=None, domain="A"):
        if domain == "AB":
            seqs = self._get_embedding(log_seqs, aug_weights=aug_weights, domain=domain)
            seqs *= self.item_emb_llm.embedding_dim ** 0.5
            seqs += self.pos_emb(positions.long())
            seqs = self.emb_dropout(seqs)
            log_feats = self.backbone(seqs, log_seqs)
        elif domain == "A":
            seqs = self._get_embedding(log_seqs, aug_weights=aug_weights, domain=domain)
            seqs *= self.item_embA.embedding_dim ** 0.5
            seqs += self.pos_embA(positions.long())
            seqs = self.emb_dropoutA(seqs)
            log_feats = self.backboneA(seqs, log_seqs)
        elif domain == "B":
            seqs = self._get_embedding(log_seqs, aug_weights=aug_weights, domain=domain)
            seqs *= self.item_embB.embedding_dim ** 0.5
            seqs += self.pos_embB(positions.long())
            seqs = self.emb_dropoutB(seqs)
            log_feats = self.backboneB(seqs, log_seqs)
        return log_feats


    def forward(self, 
                seq, pos, neg, positions,
                seqA, posA, negA, positionsA,
                seqB, posB, negB, positionsB,
                target_domain, domain_mask,
                **kwargs):
        pass 
    
    def predict(self,
                seq, item_indices, positions,
                seqA, item_indicesA, positionsA,
                seqB, item_indicesB, positionsB,
                target_domain,
                **kwargs): 
        # Inference: aug_weights can be None (default)
        aug_weights = None
        if self.use_gnn and self.aug_edge_u is not None:
             pass 

        log_feats = self.log2feats(seq, positions, aug_weights=aug_weights, domain="AB")
        final_feat = log_feats[:, -1, :] 
        item_embs = self._get_embedding(item_indices, aug_weights=aug_weights, domain="AB") 
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        
        log_featsA = self.log2feats(seqA, positionsA, aug_weights=aug_weights, domain="A")
        final_featA = log_featsA[:, -1, :] 
        item_embsA = self._get_embedding(item_indicesA, aug_weights=aug_weights, domain="A") 
        logitsA = item_embsA.matmul(final_featA.unsqueeze(-1)).squeeze(-1)

        log_featsB = self.log2feats(seqB, positionsB, aug_weights=aug_weights, domain="B")
        final_featB = log_featsB[:, -1, :] 
        item_embsB = self._get_embedding(item_indicesB, aug_weights=aug_weights, domain="B") 
        logitsB = item_embsB.matmul(final_featB.unsqueeze(-1)).squeeze(-1)

        logits[target_domain==0] += logitsA[target_domain==0]
        logits[target_domain==1] += logitsB[target_domain==1]

        return logits


class LLM4CDSR(LLM4CDSR_base):

    def __init__(self, user_num, item_num_dict, device, args):

        super().__init__(user_num, item_num_dict, device, args)
        self.args = args # Save args locally
        self.alpha = args.alpha
        self.beta = args.beta
        
        llm_user_emb = pickle.load(open("./data/{}/handled/{}.pkl".format(args.dataset, args.user_emb_file), "rb"))
        
        if not isinstance(llm_user_emb, np.ndarray):
            llm_user_emb = np.array(llm_user_emb)
            
        llm_user_emb = np.concatenate([
            np.zeros((1, llm_user_emb.shape[1])),
            llm_user_emb
        ])
        
        self.user_emb_llm = nn.Embedding.from_pretrained(torch.Tensor(llm_user_emb), padding_idx=0)
        self.user_emb_llm.weight.requires_grad = False

        self.user_adapter = nn.Sequential(
            nn.Linear(llm_user_emb.shape[1], int(llm_user_emb.shape[1] / 2)),
            nn.Linear(int(llm_user_emb.shape[1] / 2), args.hidden_size)
        )

        self.reg_loss_func = Contrastive_Loss2(tau=args.tau_reg)
        self.user_loss_func = Contrastive_Loss2(tau=args.tau)

        self.filter_init_modules.append("user_emb_llm")
        self._init_weights()
        
        if self.use_gnn:
             self.fusion_gate = nn.Sequential(
                 nn.Linear(args.hidden_size * 2, 1, bias=False),
                 nn.Sigmoid()
             )
             nn.init.xavier_uniform_(self.fusion_gate[0].weight)
             
             # [AGSL] Initialize Adaptive Graph Learner
             self.graph_learner = GraphLearner(input_dim=args.hidden_size).to(device)
             self.graph_learn_tau = getattr(args, 'graph_learn_tau', 1.0)
             self.graph_learn_weight = getattr(args, 'graph_learn_weight', 0.01)

    def _build_cross_domain_graph(self, args):
        super()._build_cross_domain_graph(args)
        # Instead of refining directly, we init candidates
        self._init_graph_structure_candidates(args)

    def _init_graph_structure_candidates(self, args):
            """
            [AGSL Upgrade] Global Semantic Graph Initialization.
            修复维度不匹配问题：临时加载原始的 1536维 Item Embedding 进行计算。
            """
            print(">> [AGSL Upgrade] Initializing Global Semantic Candidates (Batch-wise)...")
            aug_k = getattr(args, 'graph_aug_k', 2)
            
            # 确保图结构存在
            if self.edge_u is None:
                return

            cpu = torch.device('cpu')
            
            # === 修复开始：加载原始 Item Embeddings (1536 dim) ===
            try:
                # 构造原始文件路径 (假设 llm_emb_file='itm_emb_np')
                # 路径格式: ./data/amazon/handled/itm_emb_np_A.pkl
                raw_A_path = "./data/{}/handled/{}_A.pkl".format(args.dataset, args.llm_emb_file)
                raw_B_path = "./data/{}/handled/{}_B.pkl".format(args.dataset, args.llm_emb_file)
                
                emb_A = pickle.load(open(raw_A_path, "rb"))
                emb_B = pickle.load(open(raw_B_path, "rb"))
                
                # 转换为 numpy
                if not isinstance(emb_A, np.ndarray): emb_A = np.array(emb_A)
                if not isinstance(emb_B, np.ndarray): emb_B = np.array(emb_B)
                
                # 拼接 A 和 B (注意顺序必须是 A 前 B 后)
                raw_items_np = np.concatenate([emb_A, emb_B], axis=0)
                
                # 转换为 Tensor
                items_emb_static = torch.from_numpy(raw_items_np).float().to(cpu)
                
                # 添加 Padding (index 0)
                # Embedding layer 的 padding_idx=0，所以需要在第0行补全
                padding_row = torch.zeros((1, items_emb_static.shape[1]), device=cpu)
                items_emb_static = torch.cat([padding_row, items_emb_static], dim=0)
                
                print(f"   Loaded raw item embeddings for graph init: {items_emb_static.shape}")

            except Exception as e:
                print(f"   [Error] Failed to load raw item embeddings: {e}")
                print(f"   [Check] Please ensure {raw_A_path} exists and is 1536-dim.")
                return
            # === 修复结束 ===

            items_emb_static = F.normalize(items_emb_static, p=2, dim=1)
            
            # 2. 获取所有 User 的 Profile LLM Embedding (Static)
            users_emb_static = self.user_emb_llm.weight.detach().cpu() # [User_Num, 1536]
            
            # 检查维度是否一致
            if users_emb_static.shape[1] != items_emb_static.shape[1]:
                print(f"   [Error] Dimension mismatch in graph init: User {users_emb_static.shape[1]} vs Item {items_emb_static.shape[1]}")
                return

            users_emb_static = F.normalize(users_emb_static, p=2, dim=1)

            # 3. 分块计算 Top-K (防止 OOM)
            batch_size = 1024  
            num_users = self.user_num
            
            keep_u_list = []
            keep_i_list = []
            keep_scores_list = []

            # 只需要计算活跃物品（去掉 padding 0）
            items_active = items_emb_static[1:] 

            print(f"   Processing {num_users} users in batches...")
            for start_idx in tqdm(range(1, num_users + 1, batch_size)):
                end_idx = min(start_idx + batch_size, num_users + 1)
                
                # 当前 Batch 的 User Embedding
                batch_users = users_emb_static[start_idx:end_idx]
                
                # 计算相似度: [Batch, Item_Num]
                scores = torch.mm(batch_users, items_active.t())
                
                # Top-K
                vals, idxs = torch.topk(scores, k=aug_k, dim=1)
                
                # 映射回全局 ID (idxs 是相对于 items_active 的索引，所以 +1)
                batch_item_ids = idxs + 1
                
                # 生成 User ID 列表
                batch_user_ids = torch.arange(start_idx, end_idx).unsqueeze(1).expand_as(batch_item_ids)
                
                keep_u_list.append(batch_user_ids.flatten())
                keep_i_list.append(batch_item_ids.flatten())
                keep_scores_list.append(vals.flatten())

            # 4. 合并所有候选边
            keep_u = torch.cat(keep_u_list)
            keep_i = torch.cat(keep_i_list)
            keep_scores = torch.cat(keep_scores_list)

            print(f"   [AGSL] Global candidates initialized. Total edges: {len(keep_u)}")

            self.aug_edge_u = keep_u.to(self.dev)
            self.aug_edge_i = keep_i.to(self.dev)
            self.aug_prior_scores = keep_scores.to(self.dev)

    def sample_graph(self):
        """
        [AGSL] Use GraphLearner to predict edge weights for candidates.
        Returns: aug_weights [E_aug, 1], distill_loss
        """
        if self.aug_edge_u is None:
            return None, torch.tensor(0.0, device=self.dev)
        
        # Get current learned embeddings for candidates
        u_emb = self.user_gnn(self.aug_edge_u)
        i_emb = self.item_gnn(self.aug_edge_i)
        
        # Predict logits
        logits = self.graph_learner(u_emb, i_emb) # [E_aug, 1]
        
        # Gumbel-Softmax Sampling for differentiability
        logits_2d = torch.cat([torch.zeros_like(logits), logits], dim=-1) # [E, 2], ref 0 is threshold
        gumbel_out = F.gumbel_softmax(logits_2d, tau=self.graph_learn_tau, hard=True)
        
        # Weights for connection (Class 1)
        aug_weights = gumbel_out[:, 1].unsqueeze(-1) # [E, 1]
        
        # Knowledge Distillation Loss: Align learner's probability with LLM priors
        # We use MSE as a distillation loss to guide the structure learner
        probs = torch.sigmoid(logits)
        prior = self.aug_prior_scores.unsqueeze(-1).clamp(0, 1) 
        distill_loss = F.mse_loss(probs, prior)
        
        return aug_weights, distill_loss

    def calc_alignment_loss(self, item_seq_emb, item_indices, domain="AB"):
        """
        计算 ID Embedding (Behavior) 和 LLM Embedding (Semantic) 之间的对齐损失
        使用 InfoNCE Loss
        """
        # 1. 获取 LLM 语义向量 (Semantic View)
        # 注意：这里我们使用 adapter 后的向量，使其与 ID 空间维度一致
        if domain == "AB":
            llm_indices = item_indices.to(self.llm_item_device)
            semantic_view = self.item_emb_llm(llm_indices)
            semantic_view = self.adapter(semantic_view).to(self.dev)
        else:
            return torch.tensor(0.0, device=self.dev)

        # 2. 获取 ID 行为向量 (Behavior View)
        behavior_view = self._get_id_embedding_only(item_indices, domain)

        # 3. 归一化
        behavior_view = F.normalize(behavior_view, dim=-1)
        semantic_view = F.normalize(semantic_view, dim=-1)

        # 4. InfoNCE Loss
        # 正样本：同一个 Item 的 ID view 和 Semantic view
        pos_score = (behavior_view * semantic_view).sum(dim=-1) / self.args.align_tau
        pos_score = torch.exp(pos_score)
        
        # 负样本：Batch 内的其他 Item
        all_sim = torch.mm(behavior_view, semantic_view.t()) / self.args.align_tau
        all_score = torch.exp(all_sim).sum(dim=-1)
        
        # Loss = -log( pos / sum(all) )
        loss = -torch.log(pos_score / (all_score + 1e-8) + 1e-8)
        
        return loss.mean()

    def _get_id_embedding_only(self, item_indices, domain="AB"):
        # 辅助函数：只获取 ID Embedding，用于对齐
        # 对于 AB 域，我们使用 Struct-Enhanced Embedding 作为 Behavior View
        return self._get_struct_item_emb(item_indices, aug_weights=None, domain="AB")

    def forward(self, 
                seq, pos, neg, positions,
                seqA, posA, negA, positionsA,
                seqB, posB, negB, positionsB,
                target_domain, domain_mask,
                reg_A, reg_B,
                user_id,
                **kwargs):
        
        # --- [AGSL] Adaptive Graph Structure Learning Phase ---
        aug_weights = None
        graph_distill_loss = torch.tensor(0.0, device=self.dev)
        
        if self.use_gnn:
             aug_weights, graph_distill_loss = self.sample_graph()

        # --- Standard Forward using Weighted Graph ---
        
        # 1. Main Sequence
        log_feats = self.log2feats(seq, positions, aug_weights=aug_weights, domain="AB")
        pos_embs = self._get_embedding(pos, aug_weights=aug_weights, domain="AB")
        neg_embs = self._get_embedding(neg, aug_weights=aug_weights, domain="AB")

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)
        pos_labels = torch.ones(pos_logits.shape, device=self.dev)
        neg_labels = torch.zeros(neg_logits.shape, device=self.dev)
        indices = (pos != 0) 
        loss = self.loss_func(pos_logits[indices], pos_labels[indices]) + self.loss_func(neg_logits[indices], neg_labels[indices])
        loss = loss.mean()
        
        # Domain A
        log_featsA = self.log2feats(seqA, positionsA, aug_weights=aug_weights, domain="A")
        pos_embsA = self._get_embedding(posA, aug_weights=aug_weights, domain="A")
        neg_embsA = self._get_embedding(negA, aug_weights=aug_weights, domain="A")
        pos_logitsA = (log_featsA * pos_embsA).sum(dim=-1)
        neg_logitsA = (log_featsA * neg_embsA).sum(dim=-1)
        # Mix logits
        pos_logitsA[posA>0] += pos_logits[domain_mask==0]
        neg_logitsA[posA>0] += neg_logits[domain_mask==0]
        indicesA = (posA!= 0)
        lossA = self.loss_func(pos_logitsA[indicesA], torch.ones_like(pos_logitsA[indicesA])) + \
                self.loss_func(neg_logitsA[indicesA], torch.zeros_like(neg_logitsA[indicesA]))
        
        # Domain B
        log_featsB = self.log2feats(seqB, positionsB, aug_weights=aug_weights, domain="B")
        pos_embsB = self._get_embedding(posB, aug_weights=aug_weights, domain="B")
        neg_embsB = self._get_embedding(negB, aug_weights=aug_weights, domain="B")
        pos_logitsB = (log_featsB * pos_embsB).sum(dim=-1)
        neg_logitsB = (log_featsB * neg_embsB).sum(dim=-1)
        # Mix logits
        pos_logitsB[posB>0] += pos_logits[domain_mask==1]
        neg_logitsB[posB>0] += neg_logits[domain_mask==1]
        indicesB = (posB!= 0)
        lossB = self.loss_func(pos_logitsB[indicesB], torch.ones_like(pos_logitsB[indicesB])) + \
                self.loss_func(neg_logitsB[indicesB], torch.zeros_like(neg_logitsB[indicesB]))
        
        loss += (lossA.mean() + lossB.mean())

        # --- [NEW] Semantic-Behavior Alignment Loss ---
        if self.args.align_weight > 0:
            valid_pos = pos[pos > 0]
            if len(valid_pos) > 0:
                align_loss = self.calc_alignment_loss(None, valid_pos, domain="AB")
                loss += self.args.align_weight * align_loss

        # --- LLM Reg Losses ---
        reg_A = reg_A[reg_A>0]
        reg_B = reg_B[reg_B>0]
        if len(reg_A) > 0 and len(reg_B) > 0:
            reg_A_emb = self._get_embedding(reg_A, aug_weights=aug_weights, domain="AB")
            reg_B_emb = self._get_embedding(reg_B, aug_weights=aug_weights, domain="AB")
            reg_loss = self.reg_loss_func(reg_A_emb, reg_B_emb)
            loss += self.alpha * reg_loss

        final_feat = log_feats[:, -1, :]
        llm_feats = self.user_adapter(self.user_emb_llm(user_id))
        user_loss = self.user_loss_func(llm_feats, final_feat)
        loss += self.beta * user_loss
        
        # --- [AGSL] Add Graph Distillation Loss ---
        if self.use_gnn:
             loss += self.graph_learn_weight * graph_distill_loss

        return loss