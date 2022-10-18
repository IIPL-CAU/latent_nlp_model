import os
import argparse
import numpy as np
import sentencepiece as spm
from tqdm import tqdm

import datasets

def pad_add(list_, max_len: int = 300):
    ind_list = list()
    for ind_ in list_:
        if len(ind_) <= max_len:
            ind = np.zeros(max_len, dtype=np.int32)
            ind[:len(ind_)] = np.array(ind_, dtype=np.int32)
            ind_list.append(ind)
        else:
            ind_list.append(ind_[:max_len])
    return np.array(ind_list, dtype=np.int32)

def spm_tokenizing(sequence_dict: dict,  args: argparse.Namespace, domain: str ='src', src_trg_identical: bool = False):

    # 0) Path Setting
    if not os.path.exists(os.path.join(args.preprocess_path, args.data_name)):
        os.mkdir(os.path.join(args.preprocess_path, args.data_name))

    preprocess_save_path = os.path.join(args.preprocess_path, args.data_name, args.tokenizer)
    if not os.path.exists(preprocess_save_path):
        os.mkdir(preprocess_save_path)

    # 1) Pre-setting
    processed_sequences = dict()
    processed_sequences['train'] = dict()
    processed_sequences['valid'] = dict()
    processed_sequences['test'] = dict()

    if domain == 'src':
        vocab_size = args.src_vocab_size
        character_coverage = args.src_character_coverage
        max_len = args.src_max_len
    if domain == 'trg':
        vocab_size = args.trg_vocab_size
        character_coverage = args.trg_character_coverage
        max_len = args.trg_max_len

    # Make text to train vocab
    with open(f'{preprocess_save_path}/{domain}.txt', 'w') as f:
        for text in sequence_dict['train'][:10000]:
            f.write(f'{text}\n')

    if src_trg_identical:
        domain = 'src'
    else:
        spm.SentencePieceProcessor()
        spm.SentencePieceTrainer.Train(
            f'--input={preprocess_save_path}/{domain}.txt --model_type={args.sentencepiece_model} '
            f'--model_prefix={preprocess_save_path}/m_{domain}_{args.sentencepiece_model}_{vocab_size} '
            f'--vocab_size={vocab_size} --character_coverage={character_coverage} '
            f'--pad_id={args.pad_id} --unk_id={args.unk_id} --bos_id={args.bos_id} --eos_id={args.eos_id} '
            f'--split_by_whitespace=true --user_defined_symbols=[SEP]')

    vocab_list = list()
    with open(f'{preprocess_save_path}/m_{domain}_{args.sentencepiece_model}_{vocab_size}.vocab') as f:
        for line in f:
            vocab_list.append(line[:-1].split('\t')[0])

    word2id = {w: i for i, w in enumerate(vocab_list)}
    spm_src = spm.SentencePieceProcessor()
    spm_src.Load(f'{preprocess_save_path}/m_{domain}_{args.sentencepiece_model}_{vocab_size}.model')

    # Encoding
    train_src_input_ids = list(
        [args.bos_id] + spm_src.encode(
                            text, enable_sampling=True, alpha=0.1, nbest_size=-1, out_type=int) + \
        [args.eos_id] for text in sequence_dict['train']
    )
    valid_src_input_ids = list(
        [args.bos_id] + spm_src.encode(text, out_type=int) + [args.eos_id] for text in sequence_dict['valid']
    )
    test_src_input_ids = list(
        [args.bos_id] + spm_src.encode(text, out_type=int) + [args.eos_id] for text in sequence_dict['test']
    )

    # Pad token add
    processed_sequences['train']['input_ids'] = pad_add(train_src_input_ids, max_len)
    processed_sequences['valid']['input_ids'] = pad_add(valid_src_input_ids, max_len)
    processed_sequences['test']['input_ids'] = pad_add(test_src_input_ids, max_len)

    # Attention mask encoding
    processed_sequences['train']['attention_mask'] = list()
    processed_sequences['valid']['attention_mask'] = list()
    processed_sequences['test']['attention_mask'] = list()

    for ind in processed_sequences['train']['input_ids']:
        processed_sequences['train']['attention_mask'].append((ind != 0).astype(int))
    for ind in processed_sequences['valid']['input_ids']:
        processed_sequences['valid']['attention_mask'].append((ind != 0).astype(int))
    for ind in processed_sequences['test']['input_ids']:
        processed_sequences['test']['attention_mask'].append((ind != 0).astype(int))

    # Segment encoding
    # processed_sequences['train']['token_type_ids'] = list()
    # processed_sequences['valid']['token_type_ids'] = list()
    # processed_sequences['test']['token_type_ids'] = list()
    processed_sequences['train']['token_type_ids'] = None
    processed_sequences['valid']['token_type_ids'] = None
    processed_sequences['test']['token_type_ids'] = None

    # for ind in processed_sequences['train']['input_ids']:
    #     print(ind)
    #     ind = list(ind)
    #     print(ind)
    #     token_type_ids_ = [0 if i <= ind.index(4) else 1 for i in range(len(ind))]
    #     token_type_ids_ = token_type_ids_ + [0 for _ in range(args.max_len - len(ind))]
    #     processed_sequences['train']['token_type_ids'].append(token_type_ids_)
    # for ind in processed_sequences['valid']['input_ids']:
    #     ind = list(ind)
    #     token_type_ids_ = [0 if i <= ind.index(4) else 1 for i in range(len(ind))]
    #     token_type_ids_ = token_type_ids_ + [0 for _ in range(args.max_len - len(ind))]
    #     processed_sequences['valid']['token_type_ids'].append(token_type_ids_)
    # for ind in processed_sequences['test']['input_ids']:
    #     ind = list(ind)
    #     token_type_ids_ = [0 if i <= ind.index(4) else 1 for i in range(len(ind))]
    #     token_type_ids_ = token_type_ids_ + [0 for _ in range(args.max_len - len(ind))]
    #     processed_sequences['test']['token_type_ids'].append(token_type_ids_)

    return processed_sequences, word2id

def benchmark_spm_tokenizing(dataset: datasets.dataset_dict.DatasetDict,  args: argparse.Namespace, domain='src'):
    assert type(dataset) == datasets.dataset_dict.DatasetDict

    # 0) Path Setting
    if not os.path.exists(os.path.join(args.preprocess_path, args.data_name)):
        os.mkdir(os.path.join(args.preprocess_path, args.data_name))

    preprocess_save_path = os.path.join(args.preprocess_path, args.data_name, args.tokenizer)
    if not os.path.exists(preprocess_save_path):
        os.mkdir(preprocess_save_path)

    # 1) Pre-setting
    processed_sequences = dict()

    for key in dataset.keys():
        processed_sequences[key] = dict()
        for k in dataset[key].column_names:
            processed_sequences[key][k] = dict()

    vocab_size = args.src_vocab_size
    character_coverage = args.src_character_coverage
    max_len = args.src_max_len

    with open(f'{preprocess_save_path}/{args.data_name}.txt', 'w') as f:
        for key in dataset['train'].column_names:
            for text in dataset['train'][key]:
                if key in ['label','idx','start1','start2','end1','end2','span1_index','span2_index']:
                    pass
                else:
                    f.write(f'{text}\n')

    spm.SentencePieceProcessor()
    spm.SentencePieceTrainer.Train(
        f'--input={preprocess_save_path}/{args.data_name}.txt --model_type={args.sentencepiece_model} '
        f'--model_prefix={preprocess_save_path}/m_{domain}_{args.sentencepiece_model}_{vocab_size} '
        f'--vocab_size={vocab_size} --character_coverage={character_coverage} '
        f'--pad_id={args.pad_id} --unk_id={args.unk_id} --bos_id={args.bos_id} --eos_id={args.eos_id} '
        f'--split_by_whitespace=true --user_defined_symbols=[SEP]')

    vocab_list = list()
    with open(f'{preprocess_save_path}/m_{domain}_{args.sentencepiece_model}_{vocab_size}.vocab') as f:
        for line in f:
            vocab_list.append(line[:-1].split('\t')[0])

    word2id = {w: i for i, w in enumerate(vocab_list)}
    spm_src = spm.SentencePieceProcessor()
    spm_src.Load(f'{preprocess_save_path}/m_{domain}_{args.sentencepiece_model}_{vocab_size}.model')

    for phase in dataset.keys():
        for key in dataset[phase].column_names:
            if key in ['label','idx','start1','start2','end1','end2','span1_index','span2_index']:
                    processed_sequences[phase][key] = dataset[phase][key]
            else:
                if phase == 'train':
                    source = list(
                    [args.bos_id] + spm_src.encode(text, enable_sampling=True, alpha=0.1, nbest_size=-1, out_type=int) + \
                    [args.eos_id] for text in dataset[phase][key])
                else:
                    source = list(
                        [args.bos_id] + spm_src.encode(text, out_type=int) + [args.eos_id] for text in dataset[phase][key])
    
                processed_sequences[phase][key]['input_ids'] = pad_add(source, max_len)
                processed_sequences[phase][key]['attention_mask'] = list()
                for ind in processed_sequences[phase][key]['input_ids']:
                    processed_sequences[phase][key]['attention_mask'].append((ind != 0).astype(int))

    return processed_sequences, word2id
