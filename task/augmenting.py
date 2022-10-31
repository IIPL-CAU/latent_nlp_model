# Import modules
import os
import gc
import psutil
import h5py
import pickle
import logging
from tqdm import tqdm
from time import time
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
# Import PyTorch
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
# Import custom modules
from model.dataset import Seq2SeqDataset, Seq2LabelDataset, MutlimodalClassificationDataset
from model.custom_transformer.transformer import Transformer
from model.custom_plm.T5 import custom_T5
from model.custom_plm.bart import custom_Bart
from model.custom_plm.bert import custom_Bert
from optimizer.utils import shceduler_select, optimizer_select
from utils import TqdmLoggingHandler, write_log, get_tb_exp_name
from task.utils import input_to_device, label_smoothing_loss, model_save_name

def training(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #===================================#
    #==============Logging==============#
    #===================================#

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(" %(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False

    if args.use_tensorboard:
        tb_writer = SummaryWriter(os.path.join(args.tensorboard_path, get_tb_exp_name(args)))
        tb_writer.add_text('args', str(args))

    write_log(logger, 'Start training!')

    #===================================#
    #============Data Load==============#
    #===================================#

    # 1) Data open
    write_log(logger, "Load data...")
    gc.disable()

    save_path = os.path.join(args.preprocess_path, args.data_name, args.tokenizer)
    if args.tokenizer == 'spm':
        save_name = f'processed_{args.task}_{args.sentencepiece_model}_src_{args.src_vocab_size}_trg_{args.trg_vocab_size}.hdf5'
    else:
        save_name = f'processed_{args.task}.hdf5'

    with h5py.File(os.path.join(save_path, save_name), 'r') as f:
        train_src_input_ids = f.get('train_src_input_ids')[:]
        train_src_attention_mask = f.get('train_src_attention_mask')[:]
        valid_src_input_ids = f.get('valid_src_input_ids')[:]
        valid_src_attention_mask = f.get('valid_src_attention_mask')[:]
        if args.task in ['translation', 'style_transfer', 'summarization']:
            train_trg_input_ids = f.get('train_trg_input_ids')[:]
            train_trg_attention_mask = f.get('train_trg_attention_mask')[:]
            valid_trg_input_ids = f.get('valid_trg_input_ids')[:]
            valid_trg_attention_mask = f.get('valid_trg_attention_mask')[:]
        elif args.task in ['reconstruction']:
            train_trg_input_ids = f.get('train_src_input_ids')[:]
            train_trg_attention_mask = f.get('train_src_attention_mask')[:]
            valid_trg_input_ids = f.get('valid_src_input_ids')[:]
            valid_trg_attention_mask = f.get('valid_src_attention_mask')[:]
        elif args.task in ['classification']:
            train_trg_list = f.get('train_label')[:]
            valid_trg_list = f.get('valid_label')[:]
        elif args.task in ['multi-modal_classification']:
            train_src_img_path = f.get('train_src_img_path')[:]
            valid_src_img_path = f.get('valid_src_img_path')[:]
            train_trg_list = f.get('train_label')[:]
            valid_trg_list = f.get('valid_label')[:]

    with open(os.path.join(save_path, save_name[:-5] + '_word2id.pkl'), 'rb') as f:
        data_ = pickle.load(f)
        src_word2id = data_['src_word2id']
        src_vocab_num = len(src_word2id)
        src_language = data_['src_language']
        if args.task in ['translation', 'style_transfer', 'summarization']:
            trg_word2id = data_['trg_word2id']
            trg_vocab_num = len(trg_word2id)
            trg_language = data_['trg_language']
        elif args.task in ['reconstruction']:
            trg_vocab_num = src_vocab_num
        else:
            trg_vocab_num = 0
        del data_

    gc.enable()
    write_log(logger, "Finished loading data!")

    #===================================#
    #===========Train setting===========#
    #===================================#

    # 1) Model initiating
    write_log(logger, 'Instantiating model...')

    variational_mode_dict = dict()
    if args.variational:
        variational_mode_dict['variational_model'] = args.variational_model
        variational_mode_dict['variational_token_processing'] = args.variational_token_processing
        variational_mode_dict['variational_with_target'] = args.variational_with_target
        variational_mode_dict['cnn_encoder'] = args.cnn_encoder
        variational_mode_dict['cnn_decoder'] = args.cnn_decoder
        variational_mode_dict['latent_add_encoder_out'] = args.latent_add_encoder_out
        variational_mode_dict['z_var'] = args.z_var
        variational_mode_dict['d_latent'] = args.d_latent