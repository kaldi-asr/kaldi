# Copyright 2021 STC-Innovation LTD (Author: Anton Mitrofanov)
from ltlm.modules.transformer_sentence_encoder import LatticeTransformerSentenceEncoder
import torch
import torch.nn as nn
from fairseq.models import (
    BaseFairseqModel,
    FairseqEncoderModel,
    register_model,
    register_model_architecture,
)

import argparse
import logging
from fairseq import utils

from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_

logger = logging.getLogger(__name__)


@register_model('ltlm')
class LTLM(BaseFairseqModel):
    """Lattice Transformer Language Model."""
    
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--encoder-layers', type=int, metavar='L',
                            help='num encoder layers')
        parser.add_argument('--arc-embed-dim', type=int, metavar='H',
                            help='Arc embedding dimention')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='H',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='F',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='A',
                            help='num encoder attention heads')
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN')
        parser.add_argument('--max-positions', type=int,
                            help='number of positional embeddings to learn')
        parser.add_argument('--load-checkpoint-heads', action='store_true',
                            help='(re-)register and load heads when loading checkpoints')
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument('--encoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for encoder')
        # args for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
        parser.add_argument('--quant-noise-pq', type=float, metavar='D', default=0,
                            help='iterative PQ quantization noise at training time')
        parser.add_argument('--quant-noise-pq-block-size', type=int, metavar='D', default=8,
                            help='block size of quantization noise at training time')
        parser.add_argument('--quant-noise-scalar', type=float, metavar='D', default=0,
                            help='scalar quantization noise and scalar quantization at training time')
        parser.add_argument('--grad-checkpointing', action='store_true', default=False,
                            help='apply gradient checkpointing in training (https://qywu.github.io/2019/05/22/explore-gradient-checkpointing.html)')


    @classmethod
    def build_from_args(cls, args, tokenizer):
        return cls(args, tokenizer)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return cls.build_from_args(args, task.tokenizer)

    def __init__(self, args, tokenizer):
        super().__init__()
        self.args = args
        self.device = 'cpu' if args.cpu else 'cuda'
        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))

        self.sentence_encoder = LatticeTransformerSentenceEncoder(
            padding_idx=tokenizer.pad(),
            vocab_size=len(tokenizer),
            num_encoder_layers=args.encoder_layers,
            arc_embedding_dim=args.arc_embedding_dim,
            embedding_dim=args.encoder_embed_dim,
            ffn_embedding_dim=args.encoder_ffn_embed_dim,
            num_attention_heads=args.encoder_attention_heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            layerdrop=args.encoder_layerdrop,
            max_seq_len=args.max_positions,
            num_segments=0,
            offset_positions_by_padding = True,
            encoder_normalize_before=True,
            apply_bert_init=True,
            activation_fn=args.activation_fn,
            embed_scale=None,
            freeze_embeddings=False,
            n_trans_layers_to_freeze=0,
            export=False,
            traceable=False,
            q_noise=args.quant_noise_pq,
            qn_block_size=args.quant_noise_pq_block_size,
            grad_checkpointing=args.grad_checkpointing,
        )
        self.output_ff = nn.Linear(args.encoder_embed_dim, 1)
        self.output_activation = nn.Sigmoid()

    def forward(self, src_tokens, features_only=False, return_all_hiddens=False, apply_sigmoid=False, masked_tokens=None, **unused):
        """
        Args:
            src_tokens (LongTensor): input tokens of shape `(batch, src_len, 3). [word_id, state_from, state_to]`
            features_only (bool, optional): skip LM head and just return
                features. If True, the output will be of shape
                `(batch, src_len, embed_dim)`.
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            tuple:
                - the LM output of shape `(batch, src_len, vocab)`
                - a dictionary of additional data, where 'inner_states'
                  is a list of hidden states. Note that the hidden
                  states have shape `(src_len, batch, vocab)`.
        """
        x, extra = self.extract_features(src_tokens, return_all_hiddens=return_all_hiddens)
        if not features_only:
            x = self.output_layer(x)
        if apply_sigmoid:
            x = self.output_activation(x)
        return x, extra

    def extract_features(self, src_tokens, return_all_hiddens=False, **unused):
        inner_states, _ = self.sentence_encoder(
            src_tokens[:, :, 0],
            last_state_only=not return_all_hiddens, positions=src_tokens[:, :, 1:]
        )
        features = inner_states[-1].transpose(0, 1)  # T x B x C -> B x T x C
        return features, {'inner_states': inner_states if return_all_hiddens else None}

    def output_layer(self, features):
        logits = self.output_ff(features)  #self.output_activation()
        return logits

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.args.max_positions

    def get_targets(self, sample, net_output):
        return sample['target']


@register_model_architecture('ltlm', 'ltlm_base')
def base_architecture(args):
    args.max_positions = getattr(args, 'max_positions', 600)
    args.arc_embedding_dim = getattr(args, 'arc_embedding_dim', 300)
    args.encoder_layers = getattr(args, 'encoder_layers', 12)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 768)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 3072)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 12)

    args.activation_fn = getattr(args, 'activation_fn', 'gelu')
    args.pooler_activation_fn = getattr(args, 'pooler_activation_fn', 'tanh')

    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.0)
    args.pooler_dropout = getattr(args, 'pooler_dropout', 0.0)
    args.encoder_layers_to_keep = getattr(args, 'encoder_layers_to_keep', None)
    args.encoder_layerdrop = getattr(args, 'encoder_layerdrop', 0.0)



@register_model_architecture('ltlm', 'lt_small')
def ltlm_small_architecture(args):
    args.encoder_layers = getattr(args, 'encoder_layers', 8)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 816)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    base_architecture(args)


@register_model_architecture('ltlm', 'lt_ultra_small')
def ltlm_ultrasmall_architecture_ultra_small(args):
    args.encoder_layers = getattr(args, 'encoder_layers', 2)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 40)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 200)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 2)
    base_architecture(args)


@register_model_architecture('ltlm', 'lt_small6')
def ltlm_small6_architecture(args):
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 300)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 1024)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 6)
    base_architecture(args)



@register_model_architecture('ltlm', 'lt_small3')
def roberta_small3_architecture(args):
    args.encoder_layers = getattr(args, 'encoder_layers', 3)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 40)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 128)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    base_architecture(args)
