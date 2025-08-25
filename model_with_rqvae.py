import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from typing import List, Tuple, Dict, Optional, Union


# ----------------------------
# RQ-VAE 相关实现 (来自model_rqvae.py)
# ----------------------------

class Codebook(nn.Module):
    """码本模块，用于特征量化"""

    def __init__(self, num_clusters: int, codebook_dim: int, device: torch.device):
        super(Codebook, self).__init__()
        self.num_clusters = num_clusters  # 码本中的聚类中心数量
        self.codebook_dim = codebook_dim  # 每个聚类中心的维度
        self.device = device  # 运行设备
        self.tolerance = 1e-5  # K-means收敛容差
        self.kmeans_iters = 100  # K-means迭代次数

        # 初始化码本 (num_clusters, codebook_dim)
        self._codebook = nn.Parameter(torch.randn(num_clusters, codebook_dim), requires_grad=False)

    def _compute_distances(self, x: Tensor) -> Tensor:
        """计算输入特征与码本中每个聚类中心的距离"""
        # x shape: (batch_size, seq_len, codebook_dim)
        # 计算每个样本到所有聚类中心的欧氏距离
        distances = (
                torch.sum(x ** 2, dim=-1, keepdim=True)
                + torch.sum(self._codebook ** 2, dim=-1)
                - 2 * torch.matmul(x, self._codebook.t())
        )
        return distances  # shape: (batch_size, seq_len, num_clusters)

    def _assign_clusters(self, distances: Tensor) -> Tensor:
        """为每个输入特征分配最近的聚类中心"""
        # 返回每个位置的最小距离索引 (聚类ID)
        return torch.argmin(distances, dim=-1)  # shape: (batch_size, seq_len)

    def _update_codebook(self, x: Tensor, assignments: Tensor) -> Tensor:
        """根据分配结果更新码本（K-means步骤）"""
        batch_size, seq_len, dim = x.shape
        x_flat = x.view(-1, dim)  # 展平为(batch_size*seq_len, dim)
        assignments_flat = assignments.view(-1)  # 展平为(batch_size*seq_len,)

        new_codebook = torch.zeros_like(self._codebook)
        counts = torch.zeros(self.num_clusters, device=self.device)

        # 计算每个聚类的平均值作为新的聚类中心
        for i in range(self.num_clusters):
            mask = (assignments_flat == i)
            if mask.any():
                new_codebook[i] = x_flat[mask].mean(dim=0)
                counts[i] = mask.sum()

        # 对没有分配到样本的聚类中心，保留原值
        new_codebook[counts == 0] = self._codebook[counts == 0]
        return new_codebook

    def fit(self, data: Tensor) -> Tuple[Tensor, Tensor]:
        """使用K-means算法初始化码本"""
        # data shape: (num_samples, codebook_dim)
        num_samples, dim = data.shape
        data = data.to(self.device)

        # 随机从数据中选择初始聚类中心
        indices = torch.randperm(num_samples)[:self.num_clusters]
        self._codebook.data = data[indices].clone()

        # K-means迭代
        for _ in range(self.kmeans_iters):
            distances = self._compute_distances(data.unsqueeze(0))  # 添加batch维度
            assignments = self._assign_clusters(distances).squeeze(0)  # 去除batch维度
            new_codebook = self._update_codebook(data.unsqueeze(0), assignments.unsqueeze(0)).squeeze(0)

            # 检查是否收敛
            if torch.norm(new_codebook - self._codebook.data) < self.tolerance:
                break

            self._codebook.data = new_codebook

        return self._codebook.data, assignments

    def quantize(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """将输入特征量化为码本中的聚类中心"""
        # x shape: (batch_size, seq_len, codebook_dim)
        batch_size, seq_len, dim = x.shape

        # 计算距离并分配聚类
        distances = self._compute_distances(x)
        assignments = self._assign_clusters(distances)  # (batch_size, seq_len)

        # 计算量化特征（查找码本）
        quantized = F.embedding(assignments, self._codebook)  # (batch_size, seq_len, codebook_dim)

        # 计算量化损失（用于VAE训练）
        commitment_loss = F.mse_loss(quantized.detach(), x)  # 承诺损失
        codebook_loss = F.mse_loss(quantized, x.detach())  # 码本损失

        # 直通估计器：梯度从x传到quantized，但前向传播使用量化值
        quantized = x + (quantized - x).detach()

        return quantized, assignments, commitment_loss + codebook_loss

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """前向传播：量化输入特征"""
        return self.quantize(x)


class RQVAE(nn.Module):
    """残差量化变分自编码器"""

    def __init__(self, input_dim: int, num_stages: int, num_clusters: int,
                 codebook_dim: int, hidden_dim: int = 256, device: torch.device = torch.device('cpu')):
        super(RQVAE, self).__init__()
        self.input_dim = input_dim  # 输入特征维度
        self.num_stages = num_stages  # 残差量化层级
        self.num_clusters = num_clusters  # 每层级码本大小
        self.codebook_dim = codebook_dim  # 码本维度
        self.device = device

        # 编码器：将输入特征映射到潜在空间
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_stages * codebook_dim)  # 输出维度适配所有层级
        )

        # 多个码本（每层级一个）
        self.codebooks = nn.ModuleList([
            Codebook(num_clusters, codebook_dim, device)
            for _ in range(num_stages)
        ])

        # 解码器：从量化特征重构输入
        self.decoder = nn.Sequential(
            nn.Linear(num_stages * codebook_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x: Tensor) -> Tensor:
        """编码：将输入特征映射到潜在空间"""
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, input_dim = x.shape

        # 展平序列维度以便通过全连接层
        x_flat = x.view(batch_size * seq_len, input_dim)
        z = self.encoder(x_flat)  # (batch_size*seq_len, num_stages*codebook_dim)

        # 重塑回带序列维度，并按层级拆分
        z = z.view(batch_size, seq_len, self.num_stages, self.codebook_dim)
        return z  # (batch_size, seq_len, num_stages, codebook_dim)

    def decode(self, quantized_list: List[Tensor]) -> Tensor:
        """解码：从量化特征重构输入"""
        # 拼接所有层级的量化特征
        quantized = torch.cat(quantized_list, dim=-1)  # (batch_size, seq_len, num_stages*codebook_dim)
        batch_size, seq_len, total_dim = quantized.shape

        # 展平序列维度以便通过全连接层
        quantized_flat = quantized.view(batch_size * seq_len, total_dim)
        x_hat = self.decoder(quantized_flat)  # (batch_size*seq_len, input_dim)

        # 重塑回带序列维度
        return x_hat.view(batch_size, seq_len, self.input_dim)  # (batch_size, seq_len, input_dim)

    def rq(self, z: Tensor) -> Tuple[List[Tensor], List[Tensor], Tensor]:
        """残差量化：逐层量化潜在特征"""
        # z shape: (batch_size, seq_len, num_stages, codebook_dim)
        batch_size, seq_len, num_stages, codebook_dim = z.shape

        quantized_list = []  # 存储每层级的量化结果
        semantic_id_list = []  # 存储每层级的语义ID
        total_loss = 0.0  # 总量化损失

        # 初始残差为0
        residual = torch.zeros(batch_size, seq_len, codebook_dim, device=self.device)

        # 逐层量化
        for i in range(num_stages):
            # 当前层级的目标 = 潜在特征 + 上一层级的残差
            target = z[:, :, i, :] + residual

            # 量化并获取量化损失
            quantized, semantic_id, loss = self.codebooks[i](target)
            total_loss += loss

            # 更新残差（用于下一层级）
            residual = target - quantized.detach()

            # 保存结果
            quantized_list.append(quantized)
            semantic_id_list.append(semantic_id)

        return quantized_list, semantic_id_list, total_loss / num_stages  # 平均量化损失

    def fit_codebooks(self, data: Tensor) -> None:
        """使用数据拟合所有码本（K-means初始化）"""
        # data shape: (num_samples, input_dim)
        with torch.no_grad():
            # 先编码得到潜在特征
            z = self.encode(data.unsqueeze(0)).squeeze(0)  # (num_samples, num_stages, codebook_dim)

            # 为每个码本拟合数据
            for i in range(self.num_stages):
                print(f"Fitting codebook stage {i + 1}/{self.num_stages}")
                self.codebooks[i].fit(z[:, i, :])

    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor], Tensor, Tensor, Tensor]:
        """完整前向传播：编码 -> 量化 -> 解码"""
        # x shape: (batch_size, seq_len, input_dim)

        # 1. 编码
        z = self.encode(x)  # (batch_size, seq_len, num_stages, codebook_dim)

        # 2. 残差量化
        quantized_list, semantic_id_list, rq_loss = self.rq(z)

        # 3. 解码重构
        x_hat = self.decode(quantized_list)  # (batch_size, seq_len, input_dim)

        # 4. 计算损失
        recon_loss = F.mse_loss(x_hat, x)  # 重构损失
        total_loss = recon_loss + 0.25 * rq_loss  # 总损失（权重可调整）

        return x_hat, semantic_id_list, recon_loss, rq_loss, total_loss


# ----------------------------
# 推荐模型相关实现 (来自model.py，已整合RQ-VAE)
# ----------------------------

class FlashMultiHeadAttention(nn.Module):
    """多头注意力机制，支持Flash Attention加速"""

    def __init__(self, hidden_units: int, num_heads: int, dropout_rate: float):
        super(FlashMultiHeadAttention, self).__init__()
        self.hidden_units = hidden_units  # 隐藏层维度
        self.num_heads = num_heads  # 注意力头数
        self.head_dim = hidden_units // num_heads  # 每个头的维度
        self.dropout_rate = dropout_rate  # dropout率

        # 确保隐藏层维度能被头数整除
        assert hidden_units % num_heads == 0, "隐藏层维度必须能被头数整除"

        # Q、K、V的线性投影层
        self.q_linear = nn.Linear(hidden_units, hidden_units)
        self.k_linear = nn.Linear(hidden_units, hidden_units)
        self.v_linear = nn.Linear(hidden_units, hidden_units)

        # 输出线性层
        self.out_linear = nn.Linear(hidden_units, hidden_units)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, attn_mask: Optional[Tensor] = None) -> Tuple[
        Tensor, None]:
        batch_size, seq_len, _ = query.size()  # 获取输入形状

        # 计算Q、K、V（通过线性层投影）
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)

        # 重塑为多头格式：[batch_size, num_heads, seq_len, head_dim]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 若PyTorch版本支持，使用Flash Attention加速；否则用标准注意力
        if hasattr(F, 'scaled_dot_product_attention'):
            attn_output = F.scaled_dot_product_attention(
                Q, K, V,
                dropout_p=self.dropout_rate if self.training else 0.0,
                attn_mask=attn_mask.unsqueeze(1) if attn_mask is not None else None
            )
        else:
            # 标准注意力计算
            scale = (self.head_dim) ** -0.5  # 缩放因子
            scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # 注意力分数

            # 应用掩码（屏蔽padding或未来信息）
            if attn_mask is not None:
                scores.masked_fill_(attn_mask.unsqueeze(1).logical_not(), float('-inf'))

            attn_weights = F.softmax(scores, dim=-1)  # 注意力权重
            attn_weights = F.dropout(attn_weights, p=self.dropout_rate, training=self.training)
            attn_output = torch.matmul(attn_weights, V)  # 加权求和

        # 重塑回原形状并通过输出层
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_units)
        output = self.out_linear(attn_output)

        return output, None


class PointWiseFeedForward(nn.Module):
    """Transformer中的前馈网络"""

    def __init__(self, hidden_units: int, dropout_rate: float):
        super(PointWiseFeedForward, self).__init__()
        # 1D卷积（等价于逐点全连接，效率更高）
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs: Tensor) -> Tensor:
        # 输入形状：[batch_size, seq_len, hidden_units]
        # 转置为Conv1D要求的[batch_size, hidden_units, seq_len]
        outputs = self.dropout2(
            self.conv2(
                self.relu(
                    self.dropout1(
                        self.conv1(inputs.transpose(-1, -2))
                    )
                )
            )
        )
        # 转回原形状
        outputs = outputs.transpose(-1, -2)
        return outputs


class BaselineModelWithRQVAE(nn.Module):
    """整合了RQ-VAE的推荐模型"""

    def __init__(self, user_num: int, item_num: int, feat_statistics: Dict,
                 feat_types: Dict, args):
        super(BaselineModelWithRQVAE, self).__init__()
        self.user_num = user_num  # 用户数量
        self.item_num = item_num  # 物品数量
        self.dev = args.device  # 设备（CPU/GPU）
        self.norm_first = args.norm_first  # 是否先归一化再注意力
        self.maxlen = args.maxlen  # 序列最大长度
        self.args = args  # 超参数

        # 嵌入层：用户、物品、位置嵌入（padding_idx=0表示padding符号）
        self.item_emb = nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.user_emb = nn.Embedding(self.user_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(2 * args.maxlen + 1, args.hidden_units, padding_idx=0)
        self.emb_dropout = nn.Dropout(p=args.dropout_rate)  # 嵌入dropout

        # 特征相关模块
        self.sparse_emb = nn.ModuleDict()  # 稀疏特征嵌入表
        self.emb_transform = nn.ModuleDict()  # 特征线性变换

        # Transformer层：注意力层、前馈层及对应的LayerNorm
        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()

        # 初始化特征信息（按类型分组）
        self._init_feat_info(feat_statistics, feat_types)

        # 初始化RQ-VAE模型（用于高维特征量化）
        self.rqvae = RQVAE(
            input_dim=args.rqvae_input_dim,  # 高维输入特征维度
            num_stages=args.rqvae_num_stages,  # 残差量化层级
            num_clusters=args.rqvae_num_clusters,  # 每层级码本大小
            codebook_dim=args.rqvae_codebook_dim,  # 码本维度
            hidden_dim=args.rqvae_hidden_dim,  # RQ-VAE隐藏层维度
            device=args.device
        )

        # 计算用户/物品特征拼接后的维度，用于全连接层映射
        # 注意：这里已经考虑了RQ-VAE量化后的特征维度变化
        userdim = args.hidden_units * (len(self.USER_SPARSE_FEAT) + 1 + len(self.USER_ARRAY_FEAT)) + len(
            self.USER_CONTINUAL_FEAT)
        itemdim = args.hidden_units * (len(self.ITEM_SPARSE_FEAT) + 1 + len(self.ITEM_ARRAY_FEAT)) + len(
            self.ITEM_CONTINUAL_FEAT) + args.rqvae_num_stages * args.rqvae_codebook_dim

        # 用户/物品特征的全连接层（映射到隐藏维度）
        self.userdnn = nn.Linear(userdim, args.hidden_units)
        self.itemdnn = nn.Linear(itemdim, args.hidden_units)
        self.last_layernorm = nn.LayerNorm(args.hidden_units, eps=1e-8)  # 最终归一化

        # 初始化Transformer块（注意力+前馈）
        for _ in range(args.num_blocks):
            self.attention_layernorms.append(nn.LayerNorm(args.hidden_units, eps=1e-8))
            self.attention_layers.append(FlashMultiHeadAttention(args.hidden_units, args.num_heads, args.dropout_rate))
            self.forward_layernorms.append(nn.LayerNorm(args.hidden_units, eps=1e-8))
            self.forward_layers.append(PointWiseFeedForward(args.hidden_units, args.dropout_rate))

        # 初始化稀疏特征嵌入表
        for k in self.USER_SPARSE_FEAT:
            self.sparse_emb[k] = nn.Embedding(self.USER_SPARSE_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        for k in self.ITEM_SPARSE_FEAT:
            self.sparse_emb[k] = nn.Embedding(self.ITEM_SPARSE_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        for k in self.USER_ARRAY_FEAT:
            self.sparse_emb[k] = nn.Embedding(self.USER_ARRAY_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        for k in self.ITEM_ARRAY_FEAT:
            self.sparse_emb[k] = nn.Embedding(self.ITEM_ARRAY_FEAT[k] + 1, args.hidden_units, padding_idx=0)

    def _init_feat_info(self, feat_statistics: Dict, feat_types: Dict) -> None:
        """初始化特征信息，按类型分组"""
        # 用户特征
        self.USER_SPARSE_FEAT = {}  # 稀疏特征
        self.USER_CONTINUAL_FEAT = []  # 连续特征
        self.USER_ARRAY_FEAT = {}  # 数组特征

        # 物品特征
        self.ITEM_SPARSE_FEAT = {}  # 稀疏特征
        self.ITEM_CONTINUAL_FEAT = []  # 连续特征
        self.ITEM_ARRAY_FEAT = {}  # 数组特征
        self.ITEM_EMB_FEAT = {}  # 高维嵌入特征（将通过RQ-VAE处理）

        # 遍历特征类型并分组
        for feat_name, feat_type in feat_types.items():
            if feat_type.startswith('user_sparse'):
                self.USER_SPARSE_FEAT[feat_name] = feat_statistics[feat_name]['max']
            elif feat_type.startswith('user_continual'):
                self.USER_CONTINUAL_FEAT.append(feat_name)
            elif feat_type.startswith('user_array'):
                self.USER_ARRAY_FEAT[feat_name] = feat_statistics[feat_name]['max']
            elif feat_type.startswith('item_sparse'):
                self.ITEM_SPARSE_FEAT[feat_name] = feat_statistics[feat_name]['max']
            elif feat_type.startswith('item_continual'):
                self.ITEM_CONTINUAL_FEAT.append(feat_name)
            elif feat_type.startswith('item_array'):
                self.ITEM_ARRAY_FEAT[feat_name] = feat_statistics[feat_name]['max']
            elif feat_type.startswith('item_emb'):
                # 高维嵌入特征，记录其维度
                self.ITEM_EMB_FEAT[feat_name] = feat_statistics[feat_name]['dim']

    def feat2tensor(self, feature_array: Dict, feat_id: str) -> Tensor:
        """将特征数组转换为张量"""
        return torch.from_numpy(feature_array[feat_id]).to(self.dev)

    def feat2emb(self, seq: Tensor, feature_array: Dict, mask: Optional[Tensor] = None,
                 include_user: bool = False) -> Tensor:
        """将序列ID和特征转换为嵌入向量，使用RQ-VAE处理高维特征"""
        seq = seq.to(self.dev)

        if include_user:  # 若包含用户特征，分离用户和物品嵌入
            user_mask = (mask == 2).to(self.dev)  # 用户掩码（2表示用户token）
            item_mask = (mask == 1).to(self.dev)  # 物品掩码（1表示物品token）

            # 用户嵌入（仅用户位置有效）
            user_embedding = self.user_emb(user_mask * seq)
            # 物品嵌入（仅物品位置有效）
            item_embedding = self.item_emb(item_mask * seq)

            item_feat_list = [item_embedding]
            user_feat_list = [user_embedding]
        else:  # 仅物品特征
            item_embedding = self.item_emb(seq)
            item_feat_list = [item_embedding]

        # 处理所有特征类型
        all_feat_types = [
            (self.ITEM_SPARSE_FEAT, 'item_sparse', item_feat_list),
            (self.ITEM_ARRAY_FEAT, 'item_array', item_feat_list),
            (self.ITEM_CONTINUAL_FEAT, 'item_continual', item_feat_list)
        ]

        if include_user:
            all_feat_types.extend([
                (self.USER_SPARSE_FEAT, 'user_sparse', user_feat_list),
                (self.USER_ARRAY_FEAT, 'user_array', user_feat_list),
                (self.USER_CONTINUAL_FEAT, 'user_continual', user_feat_list)
            ])

        # 遍历特征，转换为嵌入并加入列表
        for feat_dict, feat_type, feat_list in all_feat_types:
            if not feat_dict:
                continue

            for k in feat_dict:
                # 特征转张量
                tensor_feature = self.feat2tensor(feature_array, k)

                if feat_type.endswith('sparse'):
                    # 稀疏特征：直接查嵌入表
                    feat_list.append(self.sparse_emb[k](tensor_feature))
                elif feat_type.endswith('array'):
                    # 数组特征：嵌入后求和
                    feat_list.append(self.sparse_emb[k](tensor_feature).sum(2))
                elif feat_type.endswith('continual'):
                    # 连续特征：升维后直接加入
                    feat_list.append(tensor_feature.unsqueeze(2))

        # 处理高维嵌入特征（使用RQ-VAE量化）
        for k in self.ITEM_EMB_FEAT:
            # 获取原始高维特征
            tensor_feature = self.feat2tensor(feature_array, k)  # shape: (batch_size, seq_len, input_dim)

            # 使用RQ-VAE进行量化
            with torch.set_grad_enabled(self.training and self.args.finetune_rqvae):
                # 编码并量化
                _, semantic_id_list, _ = self.rqvae(tensor_feature)

                # 将语义ID转换为嵌入（使用码本）
                quantized_feats = []
                for i, semantic_ids in enumerate(semantic_id_list):
                    # 从码本获取嵌入
                    quantized = F.embedding(semantic_ids, self.rqvae.codebooks[i]._codebook)
                    quantized_feats.append(quantized)

                # 拼接所有层级的量化特征
                quantized_feat = torch.cat(quantized_feats, dim=-1)

            # 将量化后的特征加入列表
            item_feat_list.append(quantized_feat)

        # 拼接所有物品特征，通过全连接层映射到隐藏维度
        all_item_emb = torch.cat(item_feat_list, dim=2)
        all_item_emb = torch.relu(self.itemdnn(all_item_emb))

        if include_user:
            # 拼接用户特征并相加
            all_user_emb = torch.cat(user_feat_list, dim=2)
            all_user_emb = torch.relu(self.userdnn(all_user_emb))
            seqs_emb = all_item_emb + all_user_emb
        else:
            seqs_emb = all_item_emb

        return seqs_emb

    def log2feats(self, user_item: Tensor, mask: Tensor, seq_feature: Dict) -> Tensor:
        """将序列转换为特征表示，通过Transformer处理"""
        # 获取序列长度
        seq_len = user_item.size(1)

        # 生成位置编码
        positions = torch.arange(seq_len, dtype=torch.long, device=self.dev)
        pos_emb = self.pos_emb(positions).unsqueeze(0)  # (1, seq_len, hidden_units)

        # 获取序列嵌入（包含用户和物品特征）
        seqs_emb = self.feat2emb(user_item, seq_feature, mask, include_user=True)

        # 加入位置嵌入并应用dropout
        seqs_emb = seqs_emb + pos_emb
        seqs_emb = self.emb_dropout(seqs_emb)

        # 生成注意力掩码（屏蔽padding）
        mask = mask.to(self.dev)
        attention_mask = mask.unsqueeze(1) * mask.unsqueeze(2)  # (batch_size, seq_len, seq_len)

        # 应用Transformer块
        for i in range(len(self.attention_layers)):
            if self.norm_first:
                # 先归一化再注意力
                seqs_emb_norm = self.attention_layernorms[i](seqs_emb)
                attn_output, _ = self.attention_layers[i](seqs_emb_norm, seqs_emb_norm, seqs_emb_norm, attention_mask)
                seqs_emb = seqs_emb + attn_output

                # 前馈网络
                seqs_emb_norm = self.forward_layernorms[i](seqs_emb)
                seqs_emb = seqs_emb + self.forward_layers[i](seqs_emb_norm)
            else:
                # 先注意力再归一化
                attn_output, _ = self.attention_layers[i](seqs_emb, seqs_emb, seqs_emb, attention_mask)
                seqs_emb = self.attention_layernorms[i](seqs_emb + attn_output)

                # 前馈网络
                seqs_emb = self.forward_layernorms[i](seqs_emb + self.forward_layers[i](seqs_emb))

        # 最终归一化
        log_feats = self.last_layernorm(seqs_emb)

        return log_feats

    def forward(self, user_item: Tensor, pos_seqs: Tensor, neg_seqs: Tensor, mask: Tensor,
                next_mask: Tensor, next_action_type: Tensor, seq_feature: Dict,
                pos_feature: Dict, neg_feature: Dict) -> Tuple[Tensor, Tensor]:
        """前向传播：计算正负样本的预测结果"""
        # 获取序列特征表示
        log_feats = self.log2feats(user_item, mask, seq_feature)

        # 损失掩码（仅物品token参与损失计算）
        loss_mask = (next_mask == 1).to(self.dev)

        # 正负样本嵌入
        pos_embs = self.feat2emb(pos_seqs, pos_feature, include_user=False)
        neg_embs = self.feat2emb(neg_seqs, neg_feature, include_user=False)

        # 点积计算logits（相似度）
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # 过滤非物品token
        pos_logits = pos_logits * loss_mask
        neg_logits = neg_logits * loss_mask

        return pos_logits, neg_logits
