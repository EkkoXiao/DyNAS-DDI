import contextlib
import logging
import os

import torch
import torch.nn as nn

from lavis.common.dist_utils import download_cached_file
from lavis.common.utils import is_url
from lavis.models.base_model import BaseModel
from lavis.models.blip2_models.Qformer import BertConfig, BertLMHeadModel
from transformers import BertTokenizer
from model.gin import GNN
from peft import LoraConfig, get_peft_model
from model.nasmodel import GraceModel
    
class Blip2Base(BaseModel):
    @classmethod
    def init_tokenizer(cls):
        tokenizer = BertTokenizer.from_pretrained('./bert_pretrained/')
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer

    def maybe_autocast(self, dtype=torch.float16):
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    @classmethod
    def init_Qformer(cls, model_name, num_query_token, graph_width, cross_attention_freq=2):
        assert model_name == 'scibert'
        
        encoder_config = BertConfig.from_pretrained('./bert_pretrained/')
        encoder_config.encoder_width = graph_width
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        
        Qformer = BertLMHeadModel.from_pretrained(
            "./bert_pretrained/", config=encoder_config
        )

        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens
    
    @classmethod
    def init_distill_Qformer(cls, model_name, num_query_token, graph_width, cross_attention_freq=2):
    
        # DistilBERT doesn't have some of the features that BERT has like cross-attention layers
        encoder_config = BertConfig.from_pretrained('./distilbert_pretrained/')
        encoder_config.dim = graph_width
        encoder_config.query_length = num_query_token
        
        # Load DistilBERT model
        Qformer = BertLMHeadModel.from_pretrained('./distilbert_pretrained/', config=encoder_config)
        
        query_tokens = nn.Parameter(torch.zeros(1, num_query_token, encoder_config.hidden_size))
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        
        return Qformer, query_tokens

    @classmethod
    def init_graph_encoder(
        cls, gin_num_layers, gin_hidden_dim, gin_drop_ratio):
        graph_encoder = GNN(
            num_layer=gin_num_layers,
            emb_dim=gin_hidden_dim,
            gnn_type='gin',
            drop_ratio=gin_drop_ratio,
            JK='last',
        )
        ckpt = torch.load('gin_pretrained/graphcl_80.pth', map_location=torch.device('cpu'))
        missing_keys, unexpected_keys = graph_encoder.load_state_dict(ckpt, strict=False)
        if len(missing_keys) or len(unexpected_keys):
            print(missing_keys)
            print(unexpected_keys)
        
        ln_graph = LayerNorm(graph_encoder.num_features)
            
        return graph_encoder, ln_graph

    @classmethod
    def init_nas_encoder(
        cls,
        input_dim,
        mol=True,
        virtual=True,
        args=None,
        use_forward=True,
    ):
        graph_encoder = GraceModel(
            input_dim,
            mol=mol,
            virtual=virtual,
            args=args,
            use_forward=use_forward,
        )
        ln_graph = LayerNorm(graph_encoder.supernet.hidden_size * (graph_encoder.supernet.num_layers + 1))
        return graph_encoder, ln_graph
        

    def load_from_pretrained(self, url_or_filename):
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]

        msg = self.load_state_dict(state_dict, strict=False)

        logging.info("load checkpoint from %s" % url_or_filename)

        return msg


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor, mask=None):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

