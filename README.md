# Analysis
> Codes for paper "***Normal vs. Adversarial: Salience-based Analysis of Adversarial Samples for Relation Extraction***"
## Requirements
- Python=3.6
- Install opennre==0.1 [here](https://github.com/thunlp/OpenNRE)
- Run `pip install -r requirements.txt` to install other dependencies
## Run
- Prepare original samples of `TACRED` and `Wiki80` datasets under `dataset` folder:
  - For the sample format, refer to the following `Data format` part.
```
dataset/
├── tacred
│   │   ├── bert/  # For analysis files generated (saliency and adversary data)
│   │   └── bertentity/  # ...
│   ├── dev.txt
│   ├── rel2id.json
│   ├── test.txt
│   └── train.txt
└── wiki80
    │   ├── bert/  # ...
    │   └── bertentity/  # ...
    ├── rel2id.json
    ├── train.txt
    └── val.txt
```
- Use `train.py` to train a model (and `test.py` for testing a model), and store the models in `model` directory:
```
$ python train.py -h
usage: train.py [-h] --model_path MODEL_PATH
                [--encoder_name {bert,bertentity}] [--restore] --train_path
                TRAIN_PATH --valid_path VALID_PATH --relation_path
                RELATION_PATH [--num_epochs NUM_EPOCHS]
                [--max_seq_len MAX_SEQ_LEN] [--batch_size BATCH_SIZE]
                [--metric {micro_f1,acc}] [--pretrain_path PRETRAIN_PATH]
$ python test.py -h
usage: test.py [-h] --model_path MODEL_PATH --test_path TEST_PATH
               --relation_path RELATION_PATH [--max_seq_len MAX_SEQ_LEN]
               [--batch_size BATCH_SIZE] [--pretrain_path PRETRAIN_PATH]
```
-(Optional) Use `server/analyse_server.py` to start the service, and fetch salience scores / adversaryial samples using POST requests:
    - Configurations like `PORT` and paths to related files can be modified in `config.py`;
    - Refer to contents in `server_tutorial.ipynb` for details of fetching saliency / adversary information.
```
$ cd server && python analyse_server.py
```
- Use `adversary_fetch.py` to generate adversary samples (use `server/filter_correct.py` to generate indices of correctly predicted samples **before this step**):
```
$ python filter_correct.py -h
usage: filter_correct.py [-h] --dataset {tacred,wiki80} --model
                         {bert,bertentity} [--max_seq_len MAX_SEQ_LEN]
$ python adversary_fetch.py -h
usage: adversary_fetch.py [-h] --dataset {tacred,wiki80} --model
                          {bert,bertentity} [--adversary {pw,tf,hf,uat,gen}]
                          [--max_seq_len MAX_SEQ_LEN] [--num_jobs NUM_JOBS]
```
- Use `saliency_fetch.py` to generate salience scores (specifically, Integrated Gradients) for samples
  - Fetch saliency information local models by default; could be set to use server by add `--mode=server` option (details of server refers to above content):
```
$ python saliency_fetch.py -h
usage: Saliency fetching tool [-h] --dataset {tacred,wiki80} --model
                              {bert,bertentity}
                              [--adversary {none,pw,tf,hf,uat,gen}]
                              [--mode {local,server}]
```
- After all the above preparations, we can go for detailed analysis. This part refers to contents in `saliency_analysis.ipynb`...
## Data format
Here we use abbreviations:
- `ig` -> `integradient gradients`
- `hf` -> `HotFlip`
- `tf` -> `TextFooler`
- `pw` -> `PWWS`
- `bert` -> `BERT-CLS` model for relation classification
- `bertentity` -> `BERT-Entity Pooling` model for relation classification (MTB)
### Sample format
- Original sample:
```json
{"token": ["Tom", "Thabane", "resigned", "in", "October", "last", "year", "to", "form", "the", "All", "Basotho", "Convention", "-LRB-", "ABC", "-RRB-", ",", "crossing", "the", "floor", "with", "17", "members", "of", "parliament", ",", "causing", "constitutional", "monarch", "King", "Letsie", "III", "to", "dissolve", "parliament", "and", "call", "the", "snap", "election", "."], "h": {"pos": [10, 13]}, "t": {"pos": [0, 2]}, "relation": "org:founded_by"}
```
- Adversarial sample:
  - `index`: index of original sample.
  - `adversary_samples`: list containing adversarial samples generated from original sample, usually 1-2 samples.
  - `predictions`: list containing predicted labels and scores (confidences) for generated samples.
```json
{"index": 0, "adversary_samples": [{"token": ["tom", "thabane", "resigned", "in", "october", "last", "year", "to", "shape", "the", "all", "basotho", "convention", "-", "lrb", "-", "abc", "-", "rrb", "-", ",", "crossing", "the", "floor", "with", "17", "members", "of", "parliament", ",", "causing", "constitutional", "monarch", "king", "letsie", "iii", "to", "dissolve", "parliament", "and", "call", "the", "snap", "election."], "h": {"pos": [10, 13]}, "t": {"pos": [0, 2]}, "relation": "org:founded_by"}], "predictions": [["org:top_members/employees", 0.46564051508903503]]}
```
- Saliency scores of samples:
  - `supervised`: list containing paired `token`-`score` list of a sample under the guidance of target label.
  - `unsupervised`: list containing paired `token`-`score` list of a sample under the guidance of preciction label, and will be `null` if `prediction==label`.
```json
{"label": "no_relation", "prediction": "no_relation", "confidence": 0.9994514584541321, "supervised": [["[CLS]", 0.11201827228069305], ["manning", 0.16019529104232788], ["was", 0.038185808807611465], ["prime", 0.10125461965799332], ["minister", 0.15581603348255157], ["in", 0.008968820795416832], ["1991", 0.06009843945503235], [",", 0.0], ["and", 0.017503196373581886], ["called", 0.06781770288944244], ["a", 0.01857049949467182], ["snap", 0.19748206436634064], ["elections", 0.29609227180480957], ["in", 0.02639009803533554], ["1995", 0.09158429503440857], ["which", 0.07849864661693573], ["he", 0.09629663825035095], ["lost", 0.22963552176952362], ["to", 0.04712187126278877], ["the", 0.025772618129849434], ["[unused0]", 0.0837450623512268], ["unc", 0.23682260513305664], ["[unused1]", 0.12174557149410248], ["after", 0.2387046217918396], ["the", 0.06034918501973152], ["party", 0.3332354426383972], ["entered", 0.36311259865760803], ["an", 0.21474991738796234], ["electoral", 0.7497870922088623], ["arrangement", 1.0], ["with", 0.25021517276763916], ["the", 0.0916648730635643], ["[unused2]", 0.17208994925022125], ["national", 0.3062470853328705], ["alliance", 0.6365000009536743], ["for", 0.11650136858224869], ["reconstruction", 0.393248587846756], ["[unused3]", 0.46095871925354004], [".", 0.13862568140029907], ["[SEP]", 0.047042131423950195]], "unsupervised": null}
```
### Dataset structure
The file organization of dataset folder.
```
dataset/
├── tacred  # name of dataset
│   ├── bert  # name of model
│   │   ├── dev_correct.index  # indices of correctly predicted dev samples
│   │   ├── dev_hf_ig.txt  # integrated gradient saliency analysis for adversarial samples generated using hotflip
│   │   ├── dev_hf.txt
│   │   ├── dev_ig.txt
│   │   ├── dev_pw_ig.txt
│   │   ├── dev_pw.txt
│   │   ├── dev_tf_ig.txt
│   │   ├── dev_tf.txt
│   │   ├── test_correct.index
│   │   ├── test_hf_ig.txt
│   │   ├── test_hf.txt
│   │   ├── test_ig.txt
│   │   ├── test_pw_ig.txt
│   │   ├── test_pw.txt
│   │   ├── test_tf_ig.txt
│   │   ├── test_tf.txt
│   │   ├── train_correct.index
│   │   ├── train_hf_ig.txt
│   │   ├── train_hf.txt
│   │   ├── train_ig.txt
│   │   ├── train_pw_ig.txt
│   │   ├── train_pw.txt
│   │   ├── train_tf_ig.txt
│   │   └── train_tf.txt
│   ├── bertentity
│   │   ├── dev_correct.index
│   │   ├── dev_hf_ig.txt
│   │   ├── dev_hf.txt
│   │   ├── dev_ig.txt
│   │   ├── dev_pw_ig.txt
│   │   ├── dev_pw.txt
│   │   ├── dev_tf_ig.txt
│   │   ├── dev_tf.txt
│   │   ├── test_correct.index
│   │   ├── test_hf_ig.txt
│   │   ├── test_hf.txt
│   │   ├── test_ig.txt
│   │   ├── test_pw_ig.txt
│   │   ├── test_pw.txt
│   │   ├── test_tf_ig.txt
│   │   ├── test_tf.txt
│   │   ├── train_correct.index
│   │   ├── train_hf_ig.txt
│   │   ├── train_hf.txt
│   │   ├── train_ig.txt
│   │   ├── train_pw_ig.txt
│   │   ├── train_pw.txt
│   │   ├── train_tf_ig.txt
│   │   └── train_tf.txt
│   ├── dev.txt
│   ├── rel2id.json
│   ├── test.txt
│   └── train.txt
└── wiki80
    ├── bert
    │   ├── train_correct.index
    │   ├── train_hf_ig.txt
    │   ├── train_hf.txt
    │   ├── train_ig.txt
    │   ├── train_pw_ig.txt
    │   ├── train_pw.txt
    │   ├── train_tf_ig.txt
    │   ├── train_tf.txt
    │   ├── val_correct.index
    │   ├── val_hf_ig.txt
    │   ├── val_hf.txt
    │   ├── val_ig.txt
    │   ├── val_pw_ig.txt
    │   ├── val_pw.txt
    │   ├── val_tf_ig.txt
    │   └── val_tf.txt
    ├── bertentity
    │   ├── train_correct.index
    │   ├── train_hf_ig.txt
    │   ├── train_hf.txt
    │   ├── train_ig.txt
    │   ├── train_pw_ig.txt
    │   ├── train_pw.txt
    │   ├── train_tf_ig.txt
    │   ├── train_tf.txt
    │   ├── val_correct.index
    │   ├── val_hf_ig.txt
    │   ├── val_hf.txt
    │   ├── val_ig.txt
    │   ├── val_pw_ig.txt
    │   ├── val_pw.txt
    │   ├── val_tf_ig.txt
    │   └── val_tf.txt
    ├── rel2id.json
    ├── train.txt
    └── val.txt
```