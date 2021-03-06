# Import PyTorch
import torch
import torch.nn as nn
# Import Huggingface
# T5
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig, BertModel
from model.custom_transformer.latent_module import Latent_module

class custom_BERT(nn.Module):
    def __init__(self, isPreTrain, d_latent, variational_mode, decoder_full_model, device):
        super().__init__()

        """
        Initialize WAE model
        
        Args:
            encoder_config (dictionary): encoder transformer's configuration
            d_latent (int): latent dimension size
            device (torch.device): 
        Returns:
            log_prob (torch.Tensor): log probability of each word 
            mean (torch.Tensor): mean of latent vector
            log_var (torch.Tensor): log variance of latent vector
            z (torch.Tensor): sampled latent vector
        """
        self.d_latent = d_latent
        self.isPreTrain = isPreTrain
        self.device = device

        self.tokenizer = BertTokenizer.from_pretrained('KETI-AIR/ke-t5-base')
        if self.isPreTrain:
            self.model1 = BertModel.from_pretrained('KETI-AIR/ke-t5-base')
        else:
            model_config = BertConfig('KETI-AIR/ke-t5-base')
            self.model1 = BertModel(config=model_config)

        self.embedding = self.model1.embeddings
        self.encoder = self.model1.encoder
        self.pooler = self.model1.pooler

        


    def forward(self, src_input_ids, src_attention_mask,
                trg_input_ids, trg_attention_mask,
                non_pad_position=None, tgt_subsqeunt_mask=None):

        # Encoder1 Forward
        encoder_out = self.encoder1_embedding(src_input_ids)
        new_attention_mask = self.model1.get_extended_attention_mask(src_attention_mask, 
                                                                     src_attention_mask.shape, self.device)
        for i in range(len(self.encoder1_model)):
            encoder_out, _ = self.encoder1_model[i](hidden_states=encoder_out, 
                                                    attention_mask=new_attention_mask)

        encoder_out = self.encoder1_final_layer_norm(encoder_out)
        encoder_out = self.encoder1_dropout(encoder_out)

        # Latent
        if self.variational_mode != 0:
            # Target sentence latent mapping
            with torch.no_grad():
                encoder_out_trg = self.encoder1_embedding(trg_input_ids)
                new_attention_mask2 = self.model1.get_extended_attention_mask(trg_attention_mask, 
                                                                             trg_attention_mask.shape, self.device)
                for i in range(len(self.encoder1_model)):
                    encoder_out_trg, _ = self.encoder1_model[i](hidden_states=encoder_out_trg, 
                                                                attention_mask=new_attention_mask2)

            encoder_out, dist_loss = self.latent_module(encoder_out, encoder_out_trg)
        else:
            dist_loss = 0

        # Encoder2 Forward
        if self.decoder_full_model:
            model_out = self.model2(inputs_embeds=encoder_out,
                                    attention_mask=src_attention_mask,
                                    decoder_input_ids=trg_input_ids,
                                    decoder_attention_mask=trg_attention_mask)
            model_out = model_out['logits']
        else:
            model_out, _ = self.decoder_model(encoder_out)
            model_out = self.lm_head(model_out)

        return model_out, dist_loss

class Discirminator_model(nn.Module):
    def __init__(self, model_type, isPreTrain, device, class_token='first_token'):
        super().__init__()

        self.model_type = model_type
        self.isPreTrain = isPreTrain
        self.class_token = class_token
        self.device = device

        if self.model_type == 'T5':
            if self.isPreTrain:
                self.D_model = T5EncoderModel.from_pretrained('t5-small')
            else:
                model_config = T5Config.from_pretrained("t5-small")
                self.D_model = T5EncoderModel(config=model_config)
            d_model = self.D_model.encoder.embed_tokens.embedding_dim
            self.linear = nn.Linear(d_model, 1)

    def forward(self, z):
        out = self.D_model(inputs_embeds=z)
        out = out['last_hidden_state']
        out = self.linear(out)

        if self.class_token == 'first_token':
            return out[:,0,:]
        elif self.class_token == 'mean_pooling':
            return out.mean(dim=1)
        elif self.class_token == 'last_token':
            return out[:,-1,:]
        else:
            raise Exception('Choose class_token in [first_token, mean_pooling, last_token]')