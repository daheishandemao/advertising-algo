import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import MyDataset
from model import BaselineModel
from model_rqvae import RQVAE
from model_rqvae import MmEmbDataset


# 格式转换
def parse_list(string):
    return [int(x) for x in string.split(',')]
def get_args():
    parser = argparse.ArgumentParser()

    # Train params
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=101, type=int)

    # Baseline Model construction
    parser.add_argument('--hidden_units', default=32, type=int)
    parser.add_argument('--num_blocks', default=1, type=int)
    parser.add_argument('--num_epochs', default=3, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--norm_first', action='store_true')

    # MMemb Feature ID
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, choices=[str(s) for s in range(81, 87)])

    # RQ-VAE Model construction
    parser.add_argument('--input_dim', default=768, type=int, help='多模态embedding维度')
    parser.add_argument('--hidden_channels', default=[512, 256], type=parse_list, help='编码器/解码器的隐藏层维度')
    parser.add_argument('--latent_dim', default=128, type=int, help='潜在空间维度')
    parser.add_argument('--num_codebooks', default=2, type=int, help='残差量化器数量')
    parser.add_argument('--codebook_size', default=[1024, 1024], type=parse_list, help='每个codebook的大小')
    parser.add_argument('--shared_codebook', action='store_false', help='是否共享codebook')
    parser.add_argument('--kmeans_method', default='kmeans', type=str, choices=['kmeans', 'bkmeans'],
                        help='K-means方法')
    parser.add_argument('--kmeans_iters', default=100, type=int, help='K-means迭代次数')
    parser.add_argument('--distances_method', default='l2', type=str, choices=['cosine', 'l2'], help='距离计算方法')
    parser.add_argument('--loss_beta', default=0.25, type=float, help='损失函数beta参数')

    args = parser.parse_args()

    return args


def train_rqvae_model(rqvae_model, args):
    """
    训练RQ-VAE模型的函数
    """
    # 确保 args.mm_emb_id 是单个值
    if isinstance(args.mm_emb_id, list):
        feature_id = args.mm_emb_id[0]  # 取第一个
    else:
        feature_id = args.mm_emb_id
    # 创建数据集和数据加载器
    dataset = MmEmbDataset(
        data_dir=os.environ.get('TRAIN_DATA_PATH'),
        feature_id=feature_id
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=256,
        collate_fn=MmEmbDataset.collate_fn,
        shuffle=True
    )

    optimizer = torch.optim.Adam(rqvae_model.parameters(), lr=1e-3)

    rqvae_model.train()
    for epoch in range(10):  # RQ-VAE训练epochs
        total_loss = 0
        for tid_batch, emb_batch in tqdm(dataloader, desc=f"RQ-VAE Epoch {epoch + 1}"):
            emb_batch = emb_batch.to(args.device)

            optimizer.zero_grad()
            x_hat, semantic_id_list, recon_loss, rqvae_loss, total_loss_batch = rqvae_model(emb_batch)

            total_loss_batch.backward()
            optimizer.step()

            total_loss += total_loss_batch.item()

        print(f"RQ-VAE Epoch {epoch + 1}, Average Loss: {total_loss / len(dataloader):.4f}")


def generate_and_process_semantic_ids(rqvae_model, data_dir, feature_id, args):
    """
    生成semantic ID并处理为baseline模型可用的格式
    """
    # 1. 生成semantic ID
    dataset = MmEmbDataset(data_dir, feature_id)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1024,
        collate_fn=MmEmbDataset.collate_fn,
        shuffle=False
    )

    semantic_id_dict = {}
    rqvae_model.eval()

    with torch.no_grad():
        for tid_batch, emb_batch in dataloader:
            emb_batch = emb_batch.to(args.device)
            # 获取semantic ID: [batch_size, num_codebooks]
            semantic_ids = rqvae_model._get_codebook(emb_batch)

            for i, tid in enumerate(tid_batch):
                tid_item = tid.item()
                # 每个codebook的ID作为独立特征
                semantic_id_list = semantic_ids[i].cpu().tolist()
                semantic_id_dict[tid_item] = semantic_id_list

    # 2. 为每个codebook创建独立的特征ID
    processed_semantic_features = {}
    vocab_sizes = {}

    for codebook_idx in range(args.num_codebooks):
        feature_name = f"semantic_id_{codebook_idx}"

        # 收集该codebook的所有semantic ID
        all_ids = [semantic_ids[codebook_idx] for semantic_ids in semantic_id_dict.values()]
        vocab_size = max(all_ids) + 1  # +1因为ID从0开始

        # 创建特征映射
        processed_semantic_features[feature_name] = {
            item_id: semantic_ids[codebook_idx]
            for item_id, semantic_ids in semantic_id_dict.items()
        }
        vocab_sizes[feature_name] = vocab_size

    return processed_semantic_features, vocab_sizes


def update_feat_config_with_semantic_ids(feat_statistics, feat_types, semantic_vocab_sizes):
    """
    将semantic ID特征添加到特征配置中
    """
    # 更新feat_statistics
    updated_feat_statistics = feat_statistics.copy()
    updated_feat_statistics.update(semantic_vocab_sizes)

    # 更新feat_types，将semantic ID作为item稀疏特征
    updated_feat_types = feat_types.copy()
    semantic_feature_names = list(semantic_vocab_sizes.keys())

    if 'item_sparse' not in updated_feat_types:
        updated_feat_types['item_sparse'] = []

    updated_feat_types['item_sparse'].extend(semantic_feature_names)

    return updated_feat_statistics, updated_feat_types


def add_semantic_ids_to_features(feat_dict, processed_semantic_features, default_value=0):
    """
    将semantic ID添加到现有的特征字典中

    Args:
        feat_dict: 原始特征字典 {item_id: {feature_name: feature_value}}
        processed_semantic_features: semantic ID特征 {feature_name: {item_id: semantic_id}}
        default_value: 没有semantic ID的item的默认值

    Returns:
        enhanced_feat_dict: 增强后的特征字典
    """
    enhanced_feat_dict = {}

    for item_id, features in feat_dict.items():
        enhanced_features = features.copy()

        # 添加semantic ID特征
        for feature_name, item_semantic_dict in processed_semantic_features.items():
            if item_id in item_semantic_dict:
                enhanced_features[feature_name] = item_semantic_dict[item_id]
            else:
                enhanced_features[feature_name] = default_value

        enhanced_feat_dict[item_id] = enhanced_features

    return enhanced_feat_dict

if __name__ == '__main__':
    Path(os.environ.get('TRAIN_LOG_PATH')).mkdir(parents=True, exist_ok=True)
    Path(os.environ.get('TRAIN_TF_EVENTS_PATH')).mkdir(parents=True, exist_ok=True)
    log_file = open(Path(os.environ.get('TRAIN_LOG_PATH'), 'train.log'), 'w')
    writer = SummaryWriter(os.environ.get('TRAIN_TF_EVENTS_PATH'))
    # global dataset
    data_path = os.environ.get('TRAIN_DATA_PATH')

    args = get_args()

    # dataset
    dataset = MyDataset(data_path, args)
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])

    # dataloader
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=128,
                                             collate_fn=MmEmbDataset.collate_fn)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=dataset.collate_fn
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=dataset.collate_fn
    )
    usernum, itemnum = dataset.usernum, dataset.itemnum
    feat_statistics, feat_types = dataset.feat_statistics, dataset.feature_types

    rqvae_model = RQVAE(
        input_dim=32,  # 输入维度
        hidden_channels=args.hidden_channels,  # 隐藏层维度列表
        latent_dim=args.latent_dim,  # 潜在空间维度
        num_codebooks=args.num_codebooks,  # 残差量化器的数量
        codebook_size=[64,64],  # 每个codebook的大小列表
        shared_codebook=args.shared_codebook,  # 是否共享codebook
        kmeans_method=args.kmeans_method,  # K-means方法，'kmeans' 或 'bkmeans'
        kmeans_iters=args.kmeans_iters,  # K-means迭代次数
        distances_method=args.distances_method,  # 距离计算方法，'cosine' 或 'l2'
        loss_beta=args.loss_beta,  # 损失函数的beta参数
        device=args.device  # 设备
    ).to(args.device)

    # 训练RQ-VAE
    train_rqvae_model(rqvae_model, args)

    # 生成semantic ID
    print("Generating semantic IDs...")
    # 确保 args.mm_emb_id 是单个值
    if isinstance(args.mm_emb_id, list):
        feature_id = args.mm_emb_id[0]  # 取第一个
    else:
        feature_id = args.mm_emb_id
    processed_semantic_features, semantic_vocab_sizes = generate_and_process_semantic_ids(
        rqvae_model,
        data_dir=os.environ.get('TRAIN_DATA_PATH'),
        feature_id=feature_id,
        args=args
    )

    # 更新特征配置
    feat_statistics, feat_types = update_feat_config_with_semantic_ids(
        feat_statistics, feat_types, semantic_vocab_sizes
    )

    # 更新特征数据
    original_feat_dict = dataset.item_feat_dict  # 加载原始特征字典
    enhanced_feat_dict = add_semantic_ids_to_features(
        original_feat_dict, processed_semantic_features
    )

    # 模型
    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except Exception:
            pass

    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0
    model.user_emb.weight.data[0, :] = 0

    for k in model.sparse_emb:
        model.sparse_emb[k].weight.data[0, :] = 0

    epoch_start_idx = 1

    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6 :]
            epoch_start_idx = int(tail[: tail.find('.')]) + 1
        except:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            raise RuntimeError('failed loading state_dicts, pls check file path!')

    bce_criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    best_val_ndcg, best_val_hr = 0.0, 0.0
    best_test_ndcg, best_test_hr = 0.0, 0.0
    T = 0.0
    t0 = time.time()
    global_step = 0
    print("Start training")
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        model.train()
        if args.inference_only:
            break
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
            seq = seq.to(args.device)
            pos = pos.to(args.device)
            neg = neg.to(args.device)
            pos_logits, neg_logits = model(
                seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
            )
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(
                neg_logits.shape, device=args.device
            )
            optimizer.zero_grad()
            indices = np.where(next_token_type == 1)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])

            log_json = json.dumps(
                {'global_step': global_step, 'loss': loss.item(), 'epoch': epoch, 'time': time.time()}
            )
            log_file.write(log_json + '\n')
            log_file.flush()
            print(log_json)

            writer.add_scalar('Loss/train', loss.item(), global_step)

            global_step += 1

            for param in model.item_emb.parameters():
                loss += args.l2_emb * torch.norm(param)
            loss.backward()
            optimizer.step()

        model.eval()
        valid_loss_sum = 0
        for step, batch in tqdm(enumerate(valid_loader), total=len(valid_loader)):
            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
            seq = seq.to(args.device)
            pos = pos.to(args.device)
            neg = neg.to(args.device)
            pos_logits, neg_logits = model(
                seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
            )
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(
                neg_logits.shape, device=args.device
            )
            indices = np.where(next_token_type == 1)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            valid_loss_sum += loss.item()
        valid_loss_sum /= len(valid_loader)
        writer.add_scalar('Loss/valid', valid_loss_sum, global_step)

        save_dir = Path(os.environ.get('TRAIN_CKPT_PATH'), f"global_step{global_step}.valid_loss={valid_loss_sum:.4f}")
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "model.pt")

    print("Done")
    writer.close()
    log_file.close()
