# Import modules
from collections import defaultdict
# Import PyTorch
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.modules.activation import MultiheadAttention

# Import custom modules
from .embedding import TransformerEmbedding
from ..latent_module.latent import Latent_module 

class Transformer(nn.Module):
    def __init__(self, task: str = 'translation', 
                 src_vocab_num: int = 8000, trg_vocab_num: int = 8000, 
                 pad_idx: int = 0, bos_idx: int = 1, eos_idx: int = 2, 
                 d_model: int = 512, d_embedding: int = 256, n_head: int = 8, 
                 dim_feedforward: int = 2048, num_common_layer: int = 10, 
                 num_encoder_layer: int = 10, num_decoder_layer: int = 10, 
                 src_max_len: int = 100, trg_max_len: int = 100, 
                 trg_emb_prj_weight_sharing: bool = False, 
                 emb_src_trg_weight_sharing: bool = False,
                 dropout: float = 0.1, embedding_dropout: float = 0.1,
                 variational: bool = True, variational_mode_dict: dict = dict(), 
                 parallel: bool = False):

        super(Transformer, self).__init__()

        # Hyper-paramter setting
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.d_model = d_model
        self.src_max_len = src_max_len
        self.trg_max_len = trg_max_len
        self.variational = variational
        self.src_vocab_num = src_vocab_num
        self.trg_vocab_num = trg_vocab_num

        # Parallel Transformer setting
        self.parallel = parallel
        if self.parallel:
            self.num_common_layer = num_common_layer
            self.num_encoder_nonparallel = num_encoder_layer - num_common_layer

        # Dropout setting
        self.dropout = nn.Dropout(dropout)

        # Source embedding part
        self.src_embedding = TransformerEmbedding(src_vocab_num, d_model, d_embedding, 
            pad_idx=self.pad_idx, max_len=self.src_max_len, dropout=embedding_dropout)

        # Target embedding part
        self.trg_embedding = TransformerEmbedding(trg_vocab_num, d_model, d_embedding,
            pad_idx=self.pad_idx, max_len=self.trg_max_len, dropout=embedding_dropout)

        # Transformer Encoder part
        self_attn = MultiheadAttention(d_model, n_head, dropout=dropout)
        self.encoders = nn.ModuleList([
            TransformerEncoderLayer(d_model, self_attn, dim_feedforward, dropout=dropout) \
                for i in range(num_encoder_layer)])

        # Transformer Decoder part
        self_attn = MultiheadAttention(d_model, n_head, dropout=dropout)
        decoder_mask_attn = MultiheadAttention(d_model, n_head, dropout=dropout)
        self.decoders = nn.ModuleList([
            TransformerDecoderLayer(d_model, self_attn, decoder_mask_attn,
                dim_feedforward, dropout=dropout) for i in range(num_decoder_layer)])

        # Target linear part
        self.trg_output_linear = nn.Linear(d_model, d_embedding)
        self.trg_output_norm = nn.LayerNorm(d_embedding, eps=1e-12)
        self.trg_output_linear2 = nn.Linear(d_embedding, trg_vocab_num)

        # Variational mode setting
        if variational:
            self.variational_model = variational_mode_dict['variational_model']
            self.variational_token_processing = variational_mode_dict['variational_token_processing']
            self.variational_with_target = variational_mode_dict['variational_with_target']
            self.cnn_encoder = variational_mode_dict['cnn_encoder']
            self.cnn_decoder = variational_mode_dict['cnn_decoder']
            self.latent_add_encoder_out = variational_mode_dict['latent_add_encoder_out']
            self.z_var = variational_mode_dict['z_var']
            self.d_latent = variational_mode_dict['d_latent']

            self.latent_module = Latent_module(d_model=self.d_hidden, d_latent=self.d_latent, 
                                               variational_model=self.variational_model, 
                                               variational_token_processing=self.variational_token_processing,
                                               variational_with_target=self.variational_with_target,
                                               cnn_encoder=self.cnn_encoder, cnn_decoder=self.cnn_decoder,
                                               latent_add_encoder_out=self.latent_add_encoder_out,
                                               z_var=self.z_var, src_max_len=src_max_len, trg_max_len=trg_max_len)
        
        # Weight sharing
        self.x_logit_scale = 1.
        if trg_emb_prj_weight_sharing:
            # Share the weight between target word embedding & last dense layer
            self.trg_output_linear2.weight = self.trg_embedding.token.weight
            self.x_logit_scale = (d_model ** -0.5)

        if emb_src_trg_weight_sharing:
            assert src_vocab_num == trg_vocab_num
            self.src_embedding.token.weight = self.trg_embedding.token.weight
            
    def forward(self, src_input_ids, src_attention_mask, src_img,
                trg_label, trg_input_ids, trg_attention_mask,
                non_pad_position=None, tgt_subsqeunt_mask=None):

        # Pre_setting for variational model and translation task
        trg_input_ids_copy = trg_input_ids.clone().detach()
        trg_input_ids = trg_input_ids[:, :-1]

        # Key padding mask setting
        src_key_padding_mask = ~src_attention_mask.bool()
        tgt_key_padding_mask = ~trg_input_ids.bool()
        tgt_key_padding_mask_ = ~trg_input_ids_copy.bool()

        # Embedding
        encoder_out = self.src_embedding(src_input_ids).transpose(0, 1) # (token, batch, d_model)
        decoder_out = self.trg_embedding(trg_input_ids).transpose(0, 1) # (token, batch, d_model)

        # Parallel Transformer
        if self.parallel:
            for i, encoder in enumerate(self.encoders):
                if i == 0:
                    encoder_out_cat = encoder(encoder_out, 
                        src_key_padding_mask=src_key_padding_mask).unsqueeze(0)
                else:
                    encoder_out_ = encoder(encoder_out_cat[-1], 
                        src_key_padding_mask=src_key_padding_mask).unsqueeze(0)
                    encoder_out_cat = torch.cat((encoder_out_cat, encoder_out_), dim=0)

            if self.variational:
                # Target sentence latent mapping
                for i, encoder in enumerate(self.encoders):
                    if i == 0:
                        if self.variational_with_target:
                            encoder_out_trg = self.trg_embedding(trg_input_ids_copy).transpose(0, 1)
                            encoder_out_trg_cat = encoder(encoder_out_trg, 
                                                        src_key_padding_mask=tgt_key_padding_mask_).unsqueeze(0)
                        else:
                            encoder_out_trg_cat = list()
                            encoder_out_trg_cat[i] = None
                        encoder_out_pre, dist_loss = self.latent_module(encoder_out_cat[i], encoder_out_trg_cat[i])
                        encoder_out_cat[i] = encoder_out_pre
                    else:
                        if self.variational_with_target:
                            encoder_out_trg_pre = encoder(encoder_out_trg_cat[-1], 
                                                        src_key_padding_mask=tgt_key_padding_mask_).unsqueeze(0)
                            encoder_out_trg_cat = torch.cat((encoder_out_trg_cat, encoder_out_trg_pre), dim=0)
                        else:
                            encoder_out_trg_cat[i] = None

                        encoder_out_pre, dist_loss_pre = self.latent_module(encoder_out, encoder_out_trg_cat[i])
                        dist_loss += dist_loss_pre
                        encoder_out_cat[i] = encoder_out_pre

            else:
                dist_loss = torch.tensor(0, dtype=torch.float)

            for i, decoder in enumerate(self.decoders):
                decoder_out = decoder(decoder_out, encoder_out_cat[i], tgt_mask=tgt_subsqeunt_mask,
                    memory_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)

        # Non-parallel Transformer
        else:
            # Encoder
            for encoder in self.encoders:
                encoder_out = encoder(encoder_out, src_key_padding_mask=src_key_padding_mask)

            # Variational
            if self.variational:
                encoder_out_trg = self.trg_embedding(trg_input_ids_copy).transpose(0, 1)
                # Target sentence latent mapping
                for encoder in self.encoders:
                    encoder_out_trg = encoder(encoder_out_trg, 
                                              src_key_padding_mask=tgt_key_padding_mask_)

                encoder_out, dist_loss = self.latent_module(encoder_out, encoder_out_trg)
            else:
                dist_loss = torch.tensor(0, dtype=torch.float)

            # Decoder
            for decoder in self.decoders:
                decoder_out = decoder(decoder_out, encoder_out, tgt_mask=tgt_subsqeunt_mask,
                    memory_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)

        decoder_out = decoder_out.transpose(0, 1).contiguous()
        if non_pad_position is not None:
            decoder_out = decoder_out[non_pad_position]

        decoder_out = self.trg_output_norm(self.dropout(F.gelu(self.trg_output_linear(decoder_out))))
        decoder_out = self.trg_output_linear2(decoder_out)
        decoder_out = decoder_out * self.x_logit_scale
        return decoder_out, dist_loss

    def generate(self, src_input_ids, src_attention_mask, beam_size, beam_alpha, repetition_penalty, device):
        # Input, output setting
        batch_size = src_input_ids.size(0)
        src_seq_size = src_input_ids.size(1)
        encoder_out_dict = defaultdict(list)
        every_batch = torch.arange(0, beam_size * batch_size, beam_size, device=device)

        # Encoding
        encoder_out = self.src_embedding(src_input_ids).transpose(0, 1) # (src_seq, batch_size, d_model)
        src_key_padding_mask = (src_input_ids == self.pad_idx) # (batch_size, src_seq)
        if self.parallel:
            for i in range(len(self.encoders)):
                encoder_out_dict[i] = self.encoders[i](encoder_out, 
                                src_key_padding_mask=src_key_padding_mask) # (src_seq, batch_size, d_model)
        else:
            for i in range(len(self.encoders)):
                encoder_out = self.encoders[i](encoder_out, 
                                src_key_padding_mask=src_key_padding_mask) # (src_seq, batch_size, d_model)

        # Expanding
        src_key_padding_mask = src_key_padding_mask.view(batch_size, 1, -1)
        src_key_padding_mask = src_key_padding_mask.repeat(1, beam_size, 1)
        src_key_padding_mask = src_key_padding_mask.view(-1, src_seq_size)
        if self.parallel:
            for i in encoder_out_dict:
                encoder_out_dict[i] = encoder_out_dict[i].view(-1, batch_size, 1, self.d_model)
                encoder_out_dict[i] = encoder_out_dict[i].repeat(1, 1, beam_size, 1)
                encoder_out_dict[i] = encoder_out_dict[i].view(src_seq_size, -1, self.d_model)
        else:
            encoder_out = encoder_out.view(-1, batch_size, 1, self.d_model)
            encoder_out = encoder_out.repeat(1, 1, beam_size, 1)
            encoder_out = encoder_out.view(src_seq_size, -1, self.d_model)

        if self.variational:
            encoder_out = self.latent_module.generate(encoder_out)

        # Scores save vector & decoding list setting
        scores_save = torch.zeros(beam_size * batch_size, 1).to(device) # (batch_size * k, 1)
        top_k_scores = torch.zeros(beam_size * batch_size, 1).to(device) # (batch_size * k, 1)
        complete_seqs = defaultdict(list)
        complete_ind = set()

        # Decoding start token setting
        seqs = torch.tensor([[self.bos_idx]], dtype=torch.long, device=device) 
        seqs = seqs.repeat(beam_size * batch_size, 1).contiguous() # (batch_size * k, 1)

        for step in range(self.trg_max_len):
            # Decoder setting
            tgt_mask = self.generate_square_subsequent_mask(seqs.size(1), device) # (out_seq)
            tgt_mask = tgt_mask.to(device, non_blocking=True)
            tgt_key_padding_mask = (seqs == self.pad_idx) # (batch_size * k, out_seq)

            # Decoding sentence
            decoder_out = self.trg_embedding(seqs).transpose(0, 1) # (out_seq, batch_size * k, d_model)
            if self.parallel:
                for i in range(len(self.decoders)):
                    decoder_out = self.decoders[i](decoder_out, encoder_out_dict[i], tgt_mask=tgt_mask, 
                                    memory_key_padding_mask=src_key_padding_mask,
                                    tgt_key_padding_mask=tgt_key_padding_mask) # (out_seq, batch_size * k, d_model)
            else:
                for i in range(len(self.decoders)):
                    decoder_out = self.decoders[i](decoder_out, encoder_out, tgt_mask=tgt_mask, 
                                    memory_key_padding_mask=src_key_padding_mask,
                                    tgt_key_padding_mask=tgt_key_padding_mask) # (out_seq, batch_size * k, d_model)

            # Score calculate
            scores = F.gelu(self.trg_output_linear(decoder_out[-1])) # (batch_size * k, d_embedding)
            scores = self.trg_output_linear2(self.trg_output_norm(scores)) # (batch_size * k, vocab_num)
            scores = F.log_softmax(scores, dim=1) # (batch_size * k, vocab_num)

            # Repetition Penalty
            if step >= 1 and repetition_penalty != 0:
                next_ix = next_word_inds.view(-1)
                for ix_ in range(len(next_ix)):
                    if scores[ix_][next_ix[ix_]] < 0:
                        scores[ix_][next_ix[ix_]] *= repetition_penalty
                    else:
                        scores[ix_][next_ix[ix_]] /= repetition_penalty

            # Add score
            scores = top_k_scores.expand_as(scores) + scores  # (batch_size * k, vocab_num)
            if step == 0:
                scores = scores[::beam_size] # (batch_size, vocab_num)
                scores[:, self.eos_idx] = float('-inf') # set eos token probability zero in first step
                top_k_scores, top_k_words = scores.topk(beam_size, 1, True, True)  # (batch_size, k) , (batch_size, k)
            else:
                top_k_scores, top_k_words = scores.view(batch_size, -1).topk(beam_size, 1, True, True)

            # Previous and Next word extract
            prev_word_inds = top_k_words // self.trg_vocab_num # (batch_size * k, out_seq)
            next_word_inds = top_k_words % self.trg_vocab_num # (batch_size * k, out_seq)
            top_k_scores = top_k_scores.view(batch_size * beam_size, -1) # (batch_size * k, out_seq)
            top_k_words = top_k_words.view(batch_size * beam_size, -1) # (batch_size * k, out_seq)
            seqs = seqs[prev_word_inds.view(-1) + every_batch.unsqueeze(1).repeat(1, beam_size).view(-1)] # (batch_size * k, out_seq)
            seqs = torch.cat([seqs, next_word_inds.view(beam_size * batch_size, -1)], dim=1) # (batch_size * k, out_seq + 1)

            # Find and Save Complete Sequences Score
            if self.eos_idx in next_word_inds:
                eos_ind = torch.where(next_word_inds.view(-1) == self.eos_idx)
                eos_ind = eos_ind[0].tolist()
                complete_ind_add = set(eos_ind) - complete_ind
                complete_ind_add = list(complete_ind_add)
                complete_ind.update(eos_ind)
                if len(complete_ind_add) > 0:
                    scores_save[complete_ind_add] = top_k_scores[complete_ind_add]
                    for ix in complete_ind_add:
                        complete_seqs[ix] = seqs[ix].tolist()

        # If eos token doesn't exist in sequence
        if 0 in scores_save:
            score_save_pos = torch.where(scores_save == 0)
            for ix in score_save_pos[0].tolist():
                complete_seqs[ix] = seqs[ix].tolist()
            scores_save[score_save_pos] = top_k_scores[score_save_pos]

        # Beam Length Normalization
        lp = torch.tensor([len(complete_seqs[i]) for i in range(batch_size * beam_size)], device=device)
        lp = (((lp + beam_size) ** beam_alpha) / ((beam_size + 1) ** beam_alpha)).unsqueeze(1)
        scores_save = scores_save / lp

        # Predicted and Label processing
        _, ind = scores_save.view(batch_size, beam_size, -1).max(1)
        ind_expand = ind.view(-1) + every_batch
        predicted = [complete_seqs[i] for i in ind_expand.tolist()]
        return predicted

    @staticmethod
    def generate_square_subsequent_mask(sz, device):
        mask = torch.tril(torch.ones(sz, sz, dtype=torch.float, device=device))
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)
        return mask

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = self_attn
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=1e-12)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.gelu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, mask_attn, dim_feedforward=2048, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = self_attn
        self.multihead_attn = mask_attn
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=1e-12)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-12)
        self.norm3 = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):

        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(F.gelu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt