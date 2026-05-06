from PIL import Image
import requests
from transformers import AutoProcessor, SiglipVisionModel
import sys
from project_path import PROJECT_PATH
sys.path.append(f"{PROJECT_PATH}/pipeline/toolbox/unisoccer")
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
from timm.models.layers import DropPath
from einops import rearrange
import torch.utils.checkpoint as checkpoint
from collections import OrderedDict
from transformers import AutoTokenizer, SiglipTextModel

class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, res_idx, d_model=768, n_head=12, drop_path=0., attn_mask=None, dropout=0., attention_type='divided_space_time', model_name="google/siglip-base-patch16-224", pretrained_layer=None):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # print(f'Droppath: {drop_path}')

        # Temporal Attention Parameters
        if attention_type == 'divided_space_time':
            self.temporal_norm1 = nn.LayerNorm(d_model)
            self.temporal_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
            self.temporal_fc = nn.Linear(d_model, d_model)
            self.register_parameter('temporal_alpha_attn', nn.Parameter(torch.tensor(0.)))

        if pretrained_layer is not None:
            self.encoder = pretrained_layer
        else:
            model = SiglipVisionModel.from_pretrained(model_name)
            self.encoder = model.vision_model.encoder.layers[res_idx]
        self.attn_mask = attn_mask

    def attention(self, x):
        return self.attn(x)[0]
    
    def temporal_attention(self, x):
        return self.temporal_attn(x, x, x)[0]

    def forward(self, x, B, T):
        # Ensure tensor shape matches expected (b*t, n, m)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        bt = x.shape[0]
        if B * T != bt and bt > 0:
            if T > 0 and bt % T == 0:
                B = bt // T
            elif B > 0 and bt % B == 0:
                T = bt // B
            else:
                B = 1
                T = bt
        # divided_space_time 

        ## Temporal 
        xt = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T)
        if xt.dim() == 2:
            xt = xt.unsqueeze(1)
        elif xt.dim() > 3:
            xt = xt.view(xt.shape[0], -1, xt.shape[-1])
        res_temporal = self.drop_path(self.temporal_attention(self.temporal_norm1(xt)))
        res_temporal = rearrange(res_temporal, '(b n) t m -> (b t) n m', b=B, t=T)
        res_temporal = self.temporal_fc(res_temporal)
        xt = x + self.temporal_alpha_attn.tanh() * res_temporal # 180 196 768

        ## Spatial
        xs = xt # always 180 196 768
        res_spatial = self.encoder(xs, self.attn_mask)
        
        return res_spatial
    

class Timesformer(nn.Module):
    def __init__(self, width, layers, heads, model_name, drop_path=0., checkpoint_num=0, dropout=0., pretrained_layers=None):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path, layers)]
        self.resblocks = nn.ModuleList()
        for idx in range(layers):
            layer = pretrained_layers[idx] if pretrained_layers is not None else None
            self.resblocks.append(ResidualAttentionBlock(d_model=width, n_head=heads, res_idx=idx, drop_path=dpr[idx], dropout=dropout, model_name=model_name, pretrained_layer=layer))
        self.checkpoint_num = checkpoint_num
            
    def forward(self, x, B, T):
        for idx, blk in enumerate(self.resblocks):
            if idx < self.checkpoint_num:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, B, T)
        return x


class VisionTimesformer(nn.Module):
    def __init__(
        self, output_dim=768, num_frames=30, 
        input_resolution = 224, patch_size = 16, width = 768,
        layers=12, heads=12,
        encoder_type = "spatial_and_temporal",
        model_name = "google/siglip-base-patch16-224"
    ):
        super().__init__()

        self.num_frames = num_frames
        model = SiglipVisionModel.from_pretrained(model_name)
        
        self.output_dim = output_dim
        self.input_resolution = input_resolution
        self.encoder_type = encoder_type
        self.patch_size = patch_size
        self.width = width
        
        if self.encoder_type == "spatial_only":
            self.vision_model = model

        elif self.encoder_type == "spatial_and_temporal":
            self.temporal_positional_embedding = nn.Parameter(torch.zeros(1, num_frames, width))

            vision_model = model.vision_model
            self.vision_model_embedding = vision_model.embeddings
            self.timesformer = Timesformer(width=width, layers=layers, heads=heads, model_name=model_name, pretrained_layers=vision_model.encoder.layers)
            self.post_layernorm = vision_model.post_layernorm
            self.head = vision_model.head


    def get_num_layers(self):
        return len(self.timesformer.resblocks)

    @torch.jit.ignore
    def no_weight_decay(self):   
        return {'temporal_positional_embedding'}

    def forward(self, x):
        B, _, T, _, _ = x.shape
        x = rearrange(x, "b c t h w -> (b t) c h w")

        if self.encoder_type == "spatial_only":
            x = self.vision_model(x)['pooler_output']
            x = rearrange(x, "(b t) m -> b t m", b=B) # 6 30 768

        elif self.encoder_type == "spatial_and_temporal":
            x = self.vision_model_embedding(x) # 180 196 768
            x = rearrange(x, "(b t) n m -> b n t m", b =B, t=T)
            x = x + self.temporal_positional_embedding
            x = rearrange(x, "b n t m -> (b t) n m") # 180 196 768
            x = self.timesformer(x, B, T) # 180 196 768
            x = self.post_layernorm(x)
            x = self.head(x) # 180 768
            x = rearrange(x, "(b t) m -> b t m", b=B, t=T) # 6 30 768

        return x
    

class TextEncoder(nn.Module):
    def __init__(
        self, model_name = "google/siglip-base-patch16-224"
    ):
        super().__init__()
        self.model = SiglipTextModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, sentences):
        # important: make sure to set padding="max_length" as that's how the model was trained
        inputs = self.tokenizer(sentences, padding="max_length", return_tensors="pt", truncation=True)
        inputs["input_ids"] = inputs["input_ids"].to(self.model.device)
        outputs = self.model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        pooled_output = outputs.pooler_output  # pooled (EOS token) states
        return pooled_output, last_hidden_state
