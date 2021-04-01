MAX_SEQ_LEN = 128
PORT = 9012
DEVICE = 'gpu'  # Could be 'cpu'
PRETRAIN_PATH = 'pretrain/bert-base-uncased/'
DATA_ROOT = 'dataset'
DATA_TYPES = {
    'wiki80': ['train', 'val'],
    'tacred': ['train', 'dev', 'test']
}
MODEL_PATH_DICT = {
    'wiki80': {
        'bert': 'model/wiki80/bert.pt',
        'bertentity': 'model/wiki80/bertentity.pt'
    },
    'tacred': {
        'bert': 'model/tacred/bert.pt',
        'bertentity': 'model/tacred/bertentity.pt'
    }
}
RELATION_PATH_DICT = {
    'wiki80': 'dataset/wiki80/rel2id.json',
    'tacred': 'dataset/tacred/rel2id.json'
}
