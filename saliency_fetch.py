import os
import json
import torch
import numpy as np
from requests import post
from argparse import ArgumentParser
from tqdm import tqdm
from opennre import encoder
from opennre.model import SoftmaxNN
from functools import partial
import logging

from config import *
from server.saliency_utils import *

# Set global device
if DEVICE == 'gpu':
    device = torch.device(
        'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
else:
    device = torch.device('cpu')

# Storages for hooks to fetch gradient & input
input_storage = []
gradient_storage = []


# Load model & register hooks
def load_model(_dataset_name, _model_name, _rel2id):
    # Silent unimportant log messages
    for logger_name in ['transformers.configuration_utils',
                        'transformers.modeling_utils',
                        'transformers.tokenization_utils_base', 'absl']:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    # Load model by path
    _encoders = {'bert': encoder.BERTEncoder,
                 'bertentity': encoder.BERTEntityEncoder}
    if _model_name in _encoders:
        _sentence_encoder = _encoders[_model_name](
            max_length=MAX_SEQ_LEN,
            pretrain_path=PRETRAIN_PATH,
            mask_entity=False
        )
        _model = SoftmaxNN(_sentence_encoder, len(_rel2id), _rel2id)
        _model.load_state_dict(torch.load(MODEL_PATH_DICT[_dataset_name][_model_name],
                                          map_location=device)['state_dict'])
        # input_data is an empty tuple
        # output_data is tensor (batch, seq_len, model_dim)
        _model.sentence_encoder.bert.embeddings.register_forward_hook(
            lambda module, input_data, output_data:
            input_storage.append(
                output_data.detach().squeeze(0).cpu().numpy()))  # (128, 768)
        # grad_input is a tuple containing one tensor (batch, seq_len, model_dim)
        # grad_output is a tuple containing one tensor (batch, seq_len, model_dim)
        _model.sentence_encoder.bert.embeddings.register_backward_hook(
            lambda module, grad_input, grad_output:
            gradient_storage.append(
                grad_input[0].detach().squeeze(0).cpu().numpy()))  # (128, 768)
        _model.to(device)
        _model.eval()

        return _model
    else:
        raise NotImplementedError


def get_saliency_local(model, sample, rel2id, id2rel):
    def get_saliency(unsupervised=False):
        def forward_backward(_inputs):
            _logits = sentence_encoder(*_inputs)
            _logits = model.softmax(model.fc(model.drop(_logits)))
            if unsupervised:
                # Use model's prediction as target
                _, pred = _logits.max(-1)
                target_label_id = pred[0].item()
            else:
                # Use supervised label as target
                target_label_id = rel2id[sample['relation']]
            ce = torch.nn.CrossEntropyLoss()
            loss = ce(_logits, torch.LongTensor([target_label_id]).to(device))
            loss.backward()
            return _logits[0].detach().cpu().numpy()

        # Tokenize sample
        sentence_encoder = model.sentence_encoder
        inputs = sentence_encoder.tokenize(sample)
        inputs = [i.to(device) for i in inputs]
        token_ids = inputs[0][0]
        tokens = sentence_encoder.tokenizer.convert_ids_to_tokens(token_ids)
        if '[PAD]' in tokens:
            tokens = tokens[:tokens.index('[PAD]')]

        # Process forward-backward once
        logits = forward_backward(inputs)
        pred = logits.argmax()
        confidence = float(logits[pred])
        pred_label = id2rel[pred]

        # Start calculating integrated gradients
        alpha_levels = 10
        for alpha_i in range(1, alpha_levels):
            alpha_rate = 1.0 * alpha_i / alpha_levels
            # Register new hook to modify embedding output each time
            alpha_hook = sentence_encoder.bert.embeddings.register_forward_hook(
                lambda module, input_data, output_data: output_data * alpha_rate)

            try:
                _ = forward_backward(inputs)
            except Exception as e:
                alpha_hook.remove()
                raise e

            alpha_hook.remove()
        gradient_average = np.mean(gradient_storage, axis=0)
        # Use first time's input
        saliency = gradient_average * input_storage[0]

        # Convert float32 to float, so that the dict is JSON serializable
        saliency = normalize(saliency[:len(tokens)])
        saliency = [float(score) for score in saliency]
        token_saliency = list(zip(tokens, saliency))
        word_saliency = merge_token_saliency(token_saliency)

        # Clear input / gradient storage
        input_storage.clear()
        gradient_storage.clear()

        return {'pred_label': pred_label,
                'confidence': confidence,
                'word_saliency': word_saliency}

    label = sample['relation']
    saliency_dict = {'label': label,
                     'prediction': '',
                     'confidence': 0.0,
                     'supervised': None,
                     'unsupervised': None}
    supervised_dict = get_saliency()
    saliency_dict['prediction'] = supervised_dict['pred_label']
    saliency_dict['confidence'] = supervised_dict['confidence']
    saliency_dict['supervised'] = supervised_dict['word_saliency']
    if supervised_dict['pred_label'] != label:
        unsupervised_dict = get_saliency(unsupervised=True)
        saliency_dict['unsupervised'] = unsupervised_dict['word_saliency']

    return saliency_dict


def get_saliency_server(args, sample):
    url = 'http://localhost:{}/saliency'.format(PORT)
    label = sample['relation']
    model = args.model
    dataset = args.dataset
    method = 'integrated_gradient'
    saliency_dict = {'label': label,
                     'prediction': '',
                     'confidence': 0.0,
                     'supervised': None,
                     'unsupervised': None}
    r = post(url, data=json.dumps({'model': model,
                                   'dataset': dataset,
                                   'method': method,
                                   'sample': sample}))
    supervised_dict = json.loads(r.text)
    saliency_dict['prediction'] = supervised_dict['pred_label']
    saliency_dict['confidence'] = supervised_dict['confidence']
    saliency_dict['supervised'] = supervised_dict['word_saliency']
    if supervised_dict['pred_label'] != label:
        r = post(url, data=json.dumps({'model': model,
                                       'dataset': dataset,
                                       'method': method,
                                       'sample': sample,
                                       'unsupervised': True}))
        unsupervised_dict = json.loads(r.text)
        saliency_dict['unsupervised'] = unsupervised_dict['word_saliency']

    return saliency_dict


if __name__ == "__main__":
    parser = ArgumentParser('Saliency fetching tool')
    parser.add_argument('--dataset', '-d', required=True,
                        choices=['tacred', 'wiki80'])
    parser.add_argument('--model', '-m', required=True,
                        choices=['bert', 'bertentity'])
    parser.add_argument('--adversary', '-a', default='none',
                        choices=['none', 'pw', 'tf', 'hf', 'uat', 'gen'])
    parser.add_argument('--mode', default='local', choices=['local', 'server'])

    args = parser.parse_args()

    if args.mode == 'local':
        rel2id = json.load(open(RELATION_PATH_DICT[args.dataset], 'r'))
        id2rel = {v: k for (k, v) in rel2id.items()}
        model = load_model(args.dataset, args.model, rel2id)
        saliency_func = partial(
            get_saliency_local, model=model, rel2id=rel2id, id2rel=id2rel)
    else:
        saliency_func = partial(get_saliency_server, args=args)

    if args.adversary == 'none':
        for data_type in DATA_TYPES[args.dataset]:
            input_file = os.path.join(
                DATA_ROOT, args.dataset, data_type + '.txt')
            output_file = os.path.join(
                DATA_ROOT, args.dataset, args.model, data_type + '_ig.txt')
            if os.path.exists(output_file):
                progress = len(open(output_file, 'r').readlines())
                logging.info('Restore progress from {} at index {}'.format(
                    output_file, progress))
            else:
                progress = 0
            out_handler = open(output_file, 'a')
            samples = [json.loads(line)
                       for line in open(input_file, 'r').readlines()]
            for sample in tqdm(samples[progress:],
                               desc='Fetching saliency for samples in ' + input_file):
                saliency = saliency_func(sample=sample)
                out_handler.write(json.dumps(saliency) + '\n')
    else:
        for data_type in DATA_TYPES[args.dataset]:
            input_file = os.path.join(
                DATA_ROOT, args.dataset, args.model,
                '{}_{}.txt'.format(data_type, args.adversary))
            output_file = os.path.join(
                DATA_ROOT, args.dataset, args.model,
                '{}_{}_ig.txt'.format(data_type, args.adversary))
            if os.path.exists(output_file):
                progress = max([json.loads(line)['index']
                                for line in open(output_file, 'r').readlines()])
                logging.info('Restore progress from {} at {}'.format(
                    output_file, progress))
            else:
                progress = -1
            out_handler = open(output_file, 'a')
            samples = [json.loads(line)
                       for line in open(input_file, 'r').readlines()]
            samples = [s for s in samples if s['index'] > progress]
            for sample_dict in tqdm(samples, desc='Fetching saliency for samples in ' + input_file):
                index = sample_dict['index']
                for sample in sample_dict['adversary_samples']:
                    saliency = saliency_func(sample=sample)
                    saliency['index'] = index
                    out_handler.write(json.dumps(saliency) + '\n')
