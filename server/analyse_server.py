import os
import sys
import json
import logging
import torch
from torch import nn
import numpy as np
from threading import Lock
from flask import Flask, request
import opennre
from opennre import encoder
from opennre.model import SoftmaxNN
import OpenAttack
from OpenAttack.utils.dataset import Dataset, DataInstance
import tensorflow as tf

try:
    sys.path.append('..')
    from server.saliency_utils import *
    from server.adversary_utils import *
    from config import *
except ImportError:
    pass

# Available saliency methods / adversary models
SUPPORTED_METHODS = {
    'saliency': ['gradient', 'gradient_x_input', 'integrated_gradient'],
    'adversary': ['pwws', 'textfooler', 'hotflip', 'generic']
}

################################################################################
# Global settings
# Enable this to allow tensors being converted to numpy arrays
tf.compat.v1.enable_eager_execution()
root_path = '..'

# Set global device
if DEVICE == 'gpu':
    device = torch.device(
        'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
else:
    device = torch.device('cpu')

# Set logging level
logging.basicConfig(level=logging.INFO)
# Silent unimportant log messages
for logger_name in ['transformers.configuration_utils',
                    'transformers.modeling_utils',
                    'transformers.tokenization_utils_base', 'absl']:
    logging.getLogger(logger_name).setLevel(logging.WARNING)


################################################################################
# Initialization of models
def load_model(_dataset_name, _model_name):
    # Load model by path
    _encoders = {'bert': encoder.BERTEncoder,
                 'bertentity': encoder.BERTEntityEncoder}
    if _model_name in _encoders:
        _sentence_encoder = _encoders[_model_name](
            max_length=MAX_SEQ_LEN,
            pretrain_path=os.path.join(root_path, PRETRAIN_PATH),
            mask_entity=False
        )
        _rel2id = relation_dict[_dataset_name]['rel2id']
        _model = SoftmaxNN(_sentence_encoder, len(_rel2id), _rel2id)
        _model.load_state_dict(torch.load(os.path.join(root_path,
                                                       MODEL_PATH_DICT[_dataset_name][_model_name]),
                                          map_location=device)['state_dict'])
        return _model
    else:
        raise NotImplementedError


model_dict = {}  # Models
relation_dict = {}  # Relation files
lock_dict = {}  # Store locks for models
for dataset_name in RELATION_PATH_DICT:
    rel2id = json.load(
        open(os.path.join(root_path, RELATION_PATH_DICT[dataset_name]), 'r'))
    id2rel = {v: k for k, v in rel2id.items()}
    relation_dict[dataset_name] = {}
    relation_dict[dataset_name]['rel2id'] = rel2id
    relation_dict[dataset_name]['id2rel'] = id2rel
for dataset_name in MODEL_PATH_DICT:
    model_dict[dataset_name] = {}
    lock_dict[dataset_name] = {}
    for model_name, model_path in MODEL_PATH_DICT[dataset_name].items():
        model = load_model(dataset_name, model_name)
        model.to(device)
        model.eval()
        model_dict[dataset_name][model_name] = model
        lock_dict[dataset_name][model_name] = Lock()


################################################################################
# Calculate saliency score for current sample
def get_saliency(data):
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

    # Check arguments
    model_name = data['model'].lower()
    dataset_name = data['dataset'].lower()
    method = data['method'].lower()
    sample = data['sample']
    unsupervised = data.get('unsupervised', False)
    assert method in SUPPORTED_METHODS['saliency']

    # Fetch corresponding model & relations
    rel2id = relation_dict[dataset_name]['rel2id']
    id2rel = relation_dict[dataset_name]['id2rel']
    model = model_dict[dataset_name][model_name]
    sentence_encoder = model.sentence_encoder

    # Process input data to get tokens
    inputs = sentence_encoder.tokenize(sample)
    inputs = [i.to(device) for i in inputs]
    token_ids = inputs[0][0]
    tokens = sentence_encoder.tokenizer.convert_ids_to_tokens(token_ids)

    # Truncate paddings
    if '[PAD]' in tokens:
        tokens = tokens[:tokens.index('[PAD]')]

    ########################################
    # Start critical section: this part will modify model
    lock = lock_dict[dataset_name][model_name]
    lock.acquire()

    try:
        # Register hooks to fetch gradient & input
        input_storage = []
        gradient_storage = []
        # input_data is an empty tuple
        # output_data is tensor (batch, seq_len, model_dim)
        input_hook = sentence_encoder.bert.embeddings.register_forward_hook(
            lambda module, input_data, output_data:
            input_storage.append(
                output_data.detach().squeeze(0).cpu().numpy()))  # (128, 768)
        # grad_input is a tuple containing one tensor (batch, seq_len, model_dim)
        # grad_output is a tuple containing one tensor (batch, seq_len, model_dim)
        gradient_hook = sentence_encoder.bert.embeddings.register_backward_hook(
            lambda module, grad_input, grad_output:
            gradient_storage.append(
                grad_input[0].detach().squeeze(0).cpu().numpy()))  # (128, 768)

        try:
            # Process forward-backward once
            logits = forward_backward(inputs)
            pred = logits.argmax()
            confidence = float(logits[pred])
            pred_label = id2rel[pred]

            # Calculate saliency scores
            if method == 'gradient':
                saliency = gradient_storage[0]
            elif method == 'gradient_x_input':
                saliency = gradient_storage[0] * input_storage[0]
            elif method == 'integrated_gradient':
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
            else:
                raise NotImplementedError

        except Exception as e:
            # Remove hooks & release lock before throwing exceptions
            input_hook.remove()
            gradient_hook.remove()
            raise e

        # Remove hooks
        input_hook.remove()
        gradient_hook.remove()

    except Exception as e:
        lock.release()
        raise e

    lock.release()
    # End of critical section
    ########################################

    # Convert float32 to float, so that the dict is JSON serializable
    saliency = normalize(saliency[:len(tokens)])
    saliency = [float(score) for score in saliency]
    token_saliency = list(zip(tokens, saliency))
    word_saliency = merge_token_saliency(token_saliency)

    return {'status': 1,
            'pred_label': pred_label,
            'confidence': confidence,
            'token_saliency': token_saliency,
            'word_saliency': word_saliency}


################################################################################
# Generate adversarial sample for current sample
def get_adversary(data):
    def load_attack(_victim, _method, _invoke_limit):
        # Attacker models
        _attack_models = {
            'pwws': OpenAttack.attackers.PWWSAttacker,
            'textfooler': OpenAttack.attackers.TextFoolerAttacker,
            'hotflip': OpenAttack.attackers.HotFlipAttacker,
            'generic': OpenAttack.attackers.GeneticAttacker
        }
        _options = {"success_rate": False, "fluency": False, "mistake": False, "semantic": False, "levenstein": False,
                    "word_distance": False, "modification_rate": False, "running_time": False, "progress_bar": False,
                    "invoke_limit": _invoke_limit, "average_invoke": False}
        if _method not in _attack_models:
            raise NotImplementedError
        if _method == 'fd':
            _attacker = _attack_models[_method]()
        else:
            _attacker = _attack_models[_method](
                skip_words=['unused0', 'unused1', 'unused2', 'unused3'])
        _attack_eval = OpenAttack.attack_evals.InvokeLimitedAttackEval(
            _attacker, _victim, **_options)
        return _attack_eval

    # Check arguments
    model_name = data['model'].lower()
    dataset_name = data['dataset'].lower()
    method = data['method'].lower()
    sample = data['sample']
    invoke_limit = data.get('invoke_limit', 700)
    assert method in SUPPORTED_METHODS['adversary']

    # Fetch corresponding model & relations
    rel2id = relation_dict[dataset_name]['rel2id']
    id2rel = relation_dict[dataset_name]['id2rel']
    model = model_dict[dataset_name][model_name]

    ########################################
    # Start critical section: this part will modify model
    lock = lock_dict[dataset_name][model_name]
    lock.acquire()

    try:
        # Build victim model
        victim = REClassifier(model, rel2id, id2rel, device)

        # Check whether the model could predict the label correctly
        # If not, this sample shouldn't be attacked
        pred_label, _ = victim.infer(sample)
        if pred_label != sample['relation']:
            return {'status': -1,
                    'message': 'Model fails to predict correctly on original sample'}

        # Convert sample to data instance
        origin = sample2data(sample, rel2id)
        # Set the label and entities so that the victim could
        # detect and react to changes in entities, which shouldn't be changed
        victim.current_label = origin.y
        victim.current_entities = get_entities(origin.x)

        # Start attack
        attack_eval = load_attack(victim, method, invoke_limit)
        adv_data = attack_eval.generate_adv([origin])
    except Exception as e:
        lock.release()
        raise e

    lock.release()
    # End of critical section
    ########################################

    raw_sample_list = dataset2sample(adv_data, id2rel)
    sample_list = []
    pred_list = []
    for sample in raw_sample_list:
        pred_label, score = victim.infer(sample)
        if pred_label != sample['relation']:
            sample_list.append(sample)
            pred_list.append((pred_label, score))

    return {'status': min(len(sample_list), 1),
            'adversary_samples': sample_list,
            'predictions': pred_list}


################################################################################
# App initialization and starting...
app = Flask(__name__)


@app.route('/saliency', methods=['POST'])
def saliency():
    try:
        data = json.loads(str(request.data, encoding='utf-8'))
        return json.dumps(get_saliency(data))
    except Exception as e:
        logging.error(str(e))
        return json.dumps({'status': -1, 'message': str(e)})


@app.route('/adversary', methods=['POST'])
def adversary():
    try:
        data = json.loads(str(request.data, encoding='utf-8'))
        return json.dumps(get_adversary(data))
    except Exception as e:
        logging.error(str(e))
        return json.dumps({'status': -1, 'message': str(e)})


app.run(host='0.0.0.0', port=PORT)
