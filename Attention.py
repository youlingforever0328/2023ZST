from re import X
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.autograd import Function
import os
import copy

import torchaudio


# override the timm package to relax the input shape constraint.
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.proj_q = nn.Linear(embed_dim, embed_dim, bias=True)
        self.proj_k = nn.Linear(embed_dim, embed_dim, bias=True)
        self.proj_v = nn.Linear(embed_dim, embed_dim, bias=True)
        self.proj_out = nn.Sequential(
                            nn.Linear(embed_dim, embed_dim, bias=False),
                            nn.Dropout(dropout))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=False):
        B, N, _ = x.shape
        q = self.proj_q(x).view(B, N, self.heads, self.head_dim).permute([0, 2, 1, 3])
        k = self.proj_k(x).view(B, N, self.heads, self.head_dim).permute([0, 2, 1, 3])
        v = self.proj_v(x).view(B, N, self.heads, self.head_dim).permute([0, 2, 1, 3])
        product = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # print(product.shape)
        if mask:
            mask = torch.zeros(B, self.heads, N, N, dtype=torch.int8, device=x.device)
            mask[:, :, 0, -1] = torch.ones(product.shape[0], product.shape[1], dtype=torch.int8, device=x.device)
            mask[:, :, -1, 0] = torch.ones(product.shape[0], product.shape[1], dtype=torch.int8, device=x.device)
            product = product.masked_fill(mask>0, -1e9) # Mask

        weights = F.softmax(product, dim=-1)
        weights = self.dropout(weights)
        out = torch.matmul(weights, v)

        # combine heads
        out = out.permute(0, 2, 1, 3)
        out = out.reshape(B, N, self.embed_dim)
        return self.proj_out(out)






class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)



class SelfEncoderLayer(nn.Module):
    def __init__(self, embed_dim=64, hidden_dim=64, num_heads=4):
        super().__init__()
        self.Attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.FeedForward = FeedForward(embed_dim, hidden_dim)
        self.Identity = nn.Identity()

    def forward(self, x, mask=False):
        residual = self.Identity(x)
        a = residual + self.Identity(self.Attention(self.norm1(x), mask=mask))
        residual = self.Identity(a)
        a = residual + self.Identity(self.FeedForward(self.norm2(a)))
        return a


class PositionEncoding(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, max_len=64):
        super(PositionEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        d_model = embed_dim if embed_dim%2==0 else embed_dim+1
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0, dtype=torch.float)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :x.size(2)]
        return self.dropout(x)




class EmbeddingTiny(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.ConstantPad1d((22, 22), 0),  # conv1
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=int(100 / 2.0), stride=int(100 / 16.0), bias=False),
            nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True),

            nn.ConstantPad1d((2, 2), 0),  # max p 1
            nn.MaxPool1d(kernel_size=8, stride=8),
            nn.Dropout(p=0.5),

            nn.ConstantPad1d((3, 4), 0),  # conv2
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, bias=False),
            nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True),

            nn.ConstantPad1d((3, 4), 0),  # conv3
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, bias=False),
            nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True),

            nn.ConstantPad1d((3, 4), 0),  # conv4
            nn.Conv1d(in_channels=128, out_channels=embed_dim, kernel_size=8, stride=1, bias=False),
            nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True),

            nn.ConstantPad1d((0, 1), 0),  # max p 2
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(p=0.5),
        )

        self.num_patches = 16

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute([0, 2, 1])
        return x



class TransformerBase(nn.Module):
    def __init__(self, embed_dim=128, hidden_dim=128*4, num_heads=8, layer_nums=3, psd=False):
        super().__init__()
        
        self.num_classes = 5
        self.embed_dim = embed_dim
        self.layer = layer_nums
        self.Embedding = EmbeddingTiny(embed_dim)
        self.PE = PositionEncoding(embed_dim)
        self.cls_s = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = nn.ModuleList([
            SelfEncoderLayer(embed_dim=embed_dim, hidden_dim=hidden_dim, num_heads=num_heads)
            for i in range(layer_nums)])


        # self.logits_eeg = nn.Sequential(
        #     nn.LayerNorm(embed_dim),
        #     nn.Linear(embed_dim, class_num)
        # )

        self.norm = nn.LayerNorm(embed_dim)
        self.out_dim = embed_dim

        # self.classifier = nn.Linear(self.embed_dim, self.num_classes, bias=False)

    def forward(self, x):
        embed = self.Embedding(x)
        cls_token = self.cls_s.expand(x.shape[0], -1, -1) 
        EEG_embed = torch.cat([cls_token, embed], dim=1)
        EEG_embed = self.PE(EEG_embed)

        for i, blk in enumerate(self.blocks):
            EEG_embed = blk(EEG_embed)

        output = self.norm(EEG_embed)
        return output






class TransformerBase2(nn.Module):
    def __init__(self, embed_dim=128, hidden_dim=128*4, num_heads=8, layer_nums=3, psd=False):
        super().__init__()
        
        self.num_classes = 5
        self.embed_dim = embed_dim
        self.layer = layer_nums
        self.Embedding = EmbeddingTiny(embed_dim)
        self.PE = PositionEncoding(embed_dim)
        self.cls_s = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_t = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = nn.ModuleList([
            SelfEncoderLayer(embed_dim=embed_dim, hidden_dim=hidden_dim, num_heads=num_heads)
            for i in range(layer_nums)])


        # self.logits_eeg = nn.Sequential(
        #     nn.LayerNorm(embed_dim),
        #     nn.Linear(embed_dim, class_num)
        # )

        self.norm = nn.LayerNorm(embed_dim)
        self.out_dim = embed_dim

        # self.classifier = nn.Linear(self.embed_dim, self.num_classes, bias=False)

    def forward(self, x, mask=True):
        embed = self.Embedding(x)
        cls_tokens = self.cls_s.expand(x.shape[0], -1, -1) 
        cls_tokent = self.cls_t.expand(x.shape[0], -1, -1) 
        EEG_embed = torch.cat([cls_tokens, embed, cls_tokent], dim=1)
        EEG_embed = self.PE(EEG_embed)

        for i, blk in enumerate(self.blocks):
            EEG_embed = blk(EEG_embed, mask=mask)

        output = self.norm(EEG_embed)
        return output







if __name__ == '__main__':
    from torchsummary import summary
    import os
    # from torchvision import models

    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    net = TransformerBase2()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
