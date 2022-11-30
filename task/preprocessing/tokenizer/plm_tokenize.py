import argparse
from transformers import  BertTokenizer, BartTokenizerFast, T5Tokenizer

from datasets import load_dataset

def plm_tokenizing(sequence_dict: dict, args: argparse.Namespace, 
                    domain: str = 'src', language: str = 'en'):

    # 1) Pre-setting
    processed_sequences = dict()
    processed_sequences['train'] = dict()
    processed_sequences['valid'] = dict()
    processed_sequences['test'] = dict()

    if domain == 'src':
        max_len = args.src_max_len
    if domain == 'trg':
        max_len = args.trg_max_len

    if args.tokenizer == 'bert':
        if language == 'en':
            tokenizer =  BertTokenizer.from_pretrained('bert-base-cased')
        elif language == 'kr':
            tokenizer =  BertTokenizer.from_pretrained('beomi/kcbert-base')
        elif language == 'de':
            tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')
        else:
            raise Exception(f'{language} language does not support')
    elif args.tokenizer == 'bart':
        if language == 'en':
            tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-large')
        elif language =='kr':
            tokenizer = BartTokenizerFast.from_pretrained('cosmoquester/bart-ko-mini')
        elif language == 'de':
            tokenizer = BartTokenizerFast.from_pretrained('Shahm/bart-german')
        else:
            raise Exception(f'{language} language does not support')
    elif args.tokenizer == 'T5':
        if language == 'en':
            tokenizer = T5Tokenizer.from_pretrained("t5-base")
        elif language == 'kr':
            tokenizer = T5Tokenizer.from_pretrained('KETI-AIR/ke-t5-base')
        elif language == 'de':
            tokenizer = T5Tokenizer.from_pretrained('t5-base')
        else:
            raise Exception(f'{language} language does not support')

    for phase in ['train', 'valid', 'test']:
        encoded_dict = \
        tokenizer(
            sequence_dict[phase],
            max_length=max_len,
            padding='max_length',
            truncation=True
        )
        processed_sequences[phase]['input_ids'] = encoded_dict['input_ids']
        processed_sequences[phase]['attention_mask'] = encoded_dict['attention_mask']
        if args.tokenizer == 'bert':
            processed_sequences[phase]['token_type_ids'] = encoded_dict['token_type_ids']

    # BART's decoder input id need to start with 'model.config.decoder_start_token_id'
    if args.tokenizer == 'bart' and domain == 'trg':
        for i in range(len(processed_sequences[phase]['input_ids'])):
            processed_sequences[phase]['input_ids'][i][0] = 2
    
    word2id = tokenizer.get_vocab()

    return processed_sequences, word2id

def benchmark_plm_tokenizing(dataset, args, domain='src', language='en'):

    # 1) Pre-setting
    processed_sequences = dict()
    
    for phase in dataset.keys():
        processed_sequences[phase] = dict()
        for key in dataset[phase].column_names:
            processed_sequences[phase][key] = dict()

    if domain == 'src':
        max_len = args.src_max_len
    if domain == 'trg':
        max_len = args.trg_max_len

    if args.tokenizer == 'bert':
        if language == 'en':
            tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    elif args.tokenizer == 'bart':
        if language == 'en':
            tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    elif args.tokenizer == 'T5':
        if language == 'en':
            tokenizer = T5Tokenizer.from_pretrained('t5-base')
    else:
        raise Exception(f'{language} language does not support')

    for phase in dataset.keys():
        for key in dataset[phase].column_names:

            # integer column
            if key in ['label','idx','start1','start2','end1','end2','span1_index','span2_index']:
                processed_sequences[phase][key] = dataset[phase][key]
            else:
                
                encoded_dict = \
                        tokenizer(
                                dataset[phase][key],
                                max_length=max_len,
                                padding='max_length',
                                truncation=True
                                )

                
                processed_sequences[phase][key]['input_ids'] = encoded_dict['input_ids']
                processed_sequences[phase][key]['attention_mask'] = encoded_dict['attention_mask']
                if args.tokenizer == 'bert':
                    processed_sequences[phase][key]['token_type_ids'] = encoded_dict['token_type_ids']


    if args.tokenizer == 'bart' and domain == 'trg':
        for i in range(len(processed_sequences[phase]['input_ids'])):
            processed_sequences[phase]['input_ids'][i][0] = 2

    word2id = tokenizer.get_vocab()

    return processed_sequences, word2id

