"""
Transformer模型实现 - 用于序列处理任务
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional, Union, Any

from .base import BaseModel


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 计算位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # 注册为缓冲区，不作为模型参数
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        添加位置编码
        
        Args:
            x: 输入张量 [seq_len, batch_size, embedding_dim]
            
        Returns:
            添加位置编码后的张量
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(BaseModel):
    """
    Transformer模型实现
    """
    
    def __init__(self, 
                 vocab_size: int,
                 d_model: int = 512, 
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 max_seq_length: int = 100):
        """
        初始化Transformer模型
        
        Args:
            vocab_size: 词汇表大小
            d_model: 模型维度
            nhead: 多头注意力头数
            num_encoder_layers: 编码器层数
            num_decoder_layers: 解码器层数
            dim_feedforward: 前馈网络维度
            dropout: Dropout比率
            max_seq_length: 最大序列长度
        """
        super().__init__()
        
        self.config = {
            'vocab_size': vocab_size,
            'd_model': d_model,
            'nhead': nhead,
            'num_encoder_layers': num_encoder_layers,
            'num_decoder_layers': num_decoder_layers,
            'dim_feedforward': dim_feedforward,
            'dropout': dropout,
            'max_seq_length': max_seq_length
        }
        
        # 模型参数
        self.model_type = 'Transformer'
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_length)
        
        # Transformer编码器和解码器
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        
        decoder_layers = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_decoder_layers)
        
        # 输出层
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        # 初始化参数
        self._init_parameters()
        
    def _init_parameters(self):
        """初始化模型参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """
        生成掩码矩阵，用于解码器的自注意力
        
        Args:
            sz: 序列长度
            
        Returns:
            掩码矩阵
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
        
    def forward(self, 
                src: torch.Tensor, 
                tgt: torch.Tensor, 
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None,
                src_padding_mask: Optional[torch.Tensor] = None,
                tgt_padding_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            src: 源序列 [src_seq_len, batch_size]
            tgt: 目标序列 [tgt_seq_len, batch_size]
            src_mask: 源序列掩码
            tgt_mask: 目标序列掩码
            src_padding_mask: 源序列填充掩码
            tgt_padding_mask: 目标序列填充掩码
            memory_mask: 记忆掩码
            memory_key_padding_mask: 记忆键填充掩码
            
        Returns:
            输出张量 [tgt_seq_len, batch_size, vocab_size]
        """
        # 如果没有提供目标掩码，自动生成
        if tgt_mask is None:
            tgt_mask = self._generate_square_subsequent_mask(tgt.size(0)).to(tgt.device)
        
        # 源序列嵌入和位置编码
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        # 目标序列嵌入和位置编码
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)
        
        # Transformer编码器
        memory = self.transformer_encoder(src, src_mask, src_padding_mask)
        
        # Transformer解码器
        output = self.transformer_decoder(tgt, memory, tgt_mask, memory_mask,
                                         tgt_padding_mask, memory_key_padding_mask)
        
        # 输出层
        output = self.output_layer(output)
        
        return output
    
    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        仅编码
        
        Args:
            src: 源序列
            src_mask: 源序列掩码
            
        Returns:
            编码后的记忆
        """
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src, src_mask)
        return memory
    
    def decode(self, 
               tgt: torch.Tensor, 
               memory: torch.Tensor, 
               tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        仅解码
        
        Args:
            tgt: 目标序列
            memory: 编码器输出的记忆
            tgt_mask: 目标序列掩码
            
        Returns:
            解码后的输出
        """
        if tgt_mask is None:
            tgt_mask = self._generate_square_subsequent_mask(tgt.size(0)).to(tgt.device)
            
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)
        output = self.transformer_decoder(tgt, memory, tgt_mask)
        output = self.output_layer(output)
        return output
    
    def generate(self, 
                src: torch.Tensor, 
                max_len: int, 
                start_symbol: int,
                end_symbol: Optional[int] = None,
                temperature: float = 1.0) -> torch.Tensor:
        """
        生成序列
        
        Args:
            src: 源序列
            max_len: 最大生成长度
            start_symbol: 起始符号ID
            end_symbol: 结束符号ID
            temperature: 采样温度
            
        Returns:
            生成的序列
        """
        self.eval()
        device = src.device
        batch_size = src.size(1)
        
        # 编码源序列
        memory = self.encode(src)
        
        # 初始化目标序列
        ys = torch.ones(1, batch_size).fill_(start_symbol).long().to(device)
        
        for i in range(max_len - 1):
            # 解码当前序列
            out = self.decode(ys, memory)
            
            # 获取最后一个时间步的预测
            prob = F.softmax(out[-1, :, :] / temperature, dim=-1)
            
            # 采样下一个词
            next_word = torch.multinomial(prob, 1).squeeze(-1)
            
            # 拼接到目标序列
            ys = torch.cat([ys, next_word.unsqueeze(0)], dim=0)
            
            # 如果所有序列都生成了结束符号，提前结束
            if end_symbol is not None and (next_word == end_symbol).all():
                break
                
        return ys