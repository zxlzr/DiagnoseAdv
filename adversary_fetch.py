import time
from multiprocessing import Queue, Process, set_start_method
from opennre import encoder

from server.adversary_utils import *
from config import *


# Enable this to allow tensors being converted to numpy arrays
tf.compat.v1.enable_eager_execution()
try:
    set_start_method('spawn')
except Exception:
    pass

logging.basicConfig(level=logging.INFO)
# Silent unimportant log messages
for logger_name in ['transformers.configuration_utils',
                    'transformers.modeling_utils',
                    'transformers.tokenization_utils_base', 'absl']:
    logging.getLogger(logger_name).setLevel(logging.ERROR)


def attack_process(idx, args, q):
    # Load victim model
    # Distribute models on devices equally
    device_id = idx % torch.cuda.device_count()  # Start from 0
    device = torch.device('cuda:' + str(device_id))

    sentence_encoder = {'bert': encoder.BERTEncoder,
                        'bertentity': encoder.BERTEntityEncoder}[args.model](
        max_length=args.max_seq_len,
        pretrain_path=PRETRAIN_PATH,
        mask_entity=False
    )
    rel2id = json.load(open(os.path.join(
        DATA_ROOT, args.dataset, 'rel2id.json'), 'r'))
    id2rel = {v: k for k, v in rel2id.items()}
    model = SoftmaxNN(sentence_encoder, len(rel2id), rel2id)
    model.load_state_dict(torch.load(MODEL_PATH_DICT[args.dataset][args.model],
                                     map_location=device)['state_dict'])
    model.eval()
    model.to(device)

    model = REClassifier(model, rel2id, id2rel, device)
    logging.info('Build model ' + str(idx) + ' on device ' + str(device_id))

    # Load attacker
    logging.info('New attacker ' + str(idx))

    # Preserve special tokens
    skip_words = ['unused0', 'unused1', 'unused2', 'unused3']

    # Attacker models
    attack_models = {
        'pw': OpenAttack.attackers.PWWSAttacker,
        'tf': OpenAttack.attackers.TextFoolerAttacker,
        'hf': OpenAttack.attackers.HotFlipAttacker,
        'uat': OpenAttack.attackers.UATAttacker,
        'gen': OpenAttack.attackers.GeneticAttacker
    }
    if args.adversary != 'uat':
        attacker = attack_models[args.adversary](skip_words=skip_words)
    else:
        attacker = attack_models[args.adversary]()

    # Build evaluation object
    options = {"success_rate": False, "fluency": False, "mistake": False, "semantic": False, "levenstein": False,
               "word_distance": False, "modification_rate": False, "running_time": False, "progress_bar": False,
               "invoke_limit": 1000, "average_invoke": True}
    attack_eval = OpenAttack.attack_evals.InvokeLimitedAttackEval(
        attacker, model, **options)

    # Generate samples one by one
    total_count, success_count, total_time = 0, 0, 0.0

    while True:
        if q.empty():
            break
        data_idx, data = q.get()
        # Save label for current sample for reference
        model.current_entities = get_entities(data.x)
        model.current_label = data.y
        start_time = time.time()
        adv_data = attack_eval.generate_adv([data])
        if len(adv_data) > 0:
            raw_sample_list = dataset2sample(adv_data, id2rel)
            sample_list = []
            prediction_list = []
            for adv_sample in raw_sample_list:
                prediction = model.infer(adv_sample)
                if prediction != id2rel[data.y]:
                    success_count += 1
                    sample_list.append(adv_sample)
                    prediction_list.append(model.infer(adv_sample))
            if sample_list:
                with open(args.output_file, 'a') as f:
                    f.write(json.dumps({'index': data_idx,
                                        'adversary_samples': sample_list,
                                        'predictions': prediction_list}) + '\n')
        cost_time = time.time() - start_time
        total_time += cost_time
        total_count += 1
        logging.info('Success:{}/{:02.2f}%, time:{:02.2f}s/{:02.2f}s, jobs:{}/{}'.format(
            len(adv_data), success_count / total_count * 100,
            cost_time, total_time / total_count,
            total_count, q.qsize()))

    logging.info('Attacker {} finished and quit.'.format(idx))

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', required=True,
                        choices=['tacred', 'wiki80'])
    parser.add_argument('--model', '-m', required=True,
                        choices=['bert', 'bertentity'])
    parser.add_argument('--adversary', '-a', default='hf',
                        choices=['pw', 'tf', 'hf', 'uat', 'gen'])
    parser.add_argument('--max_seq_len', '-l', type=int, default=128,
                        help='Maximum sequence length of bert model')
    parser.add_argument('--num_jobs', '-j', type=int, default=1,
                        help='Maximum number of parallel workers in attacking')
    args = parser.parse_args()

    logging.info('CUDA device status: {}, devices: {}'.format(
        torch.cuda.is_available(), torch.cuda.device_count()))

    # Load dataset
    for data_type in DATA_TYPES[args.dataset]:
        correct_indices = [int(line.strip()) for line in open(
            os.path.join(DATA_ROOT, args.dataset, args.model,
                         data_type + '_correct.index'), 'r').readlines()]
        samples = [json.loads(line) for line in open(
            os.path.join(DATA_ROOT, args.dataset, data_type + '.txt'), 'r').readlines()]
        args.output_file = os.path.join(DATA_ROOT, args.dataset, args.model,
                                        data_type + '_' + args.adversary + '.txt')

        # Restore progress: remove the tasks already tried in output file
        if os.path.exists(args.output_file):
            generated_samples = [json.loads(line) for line in open(
                args.output_file, 'r').readlines()]
            max_index = max([sample['index'] for sample in generated_samples])
            correct_indices = [
                idx for idx in correct_indices if idx > max_index]
            logging.info('Restore progress from index ' + str(max_index))

        correct_samples = [samples[idx] for idx in correct_indices]
        rel2id = json.load(open(
            os.path.join(DATA_ROOT, args.dataset, 'rel2id.json'), 'r'))
        id2rel = {v: k for k, v in rel2id.items()}
        dataset = sample2dataset(correct_samples, rel2id)

        # Cut dataset into mini-batches, each containing fixed number of samples
        logging.info(
            'Creating queue for dataset {}-{}'.format(args.dataset, data_type))
        logging.info('Will output to ' + args.output_file)

        queue = Queue()
        for idx, data in zip(correct_indices, dataset):
            queue.put((idx, data))
        logging.info('Total tasks: ' + str(queue.qsize()))

        # Start attacking
        logging.info('Start attacking model {} using {} attacker'.format(
            args.model, args.adversary))
        if args.num_jobs > 1:
            # Multi-process attacking
            process_list = []
            for index in range(args.num_jobs):
                p = Process(target=attack_process,
                            args=(index + 1, args, queue))
                process_list.append(p)
                p.start()
            try:
                for p in process_list:
                    p.join()
            except KeyboardInterrupt:
                logging.error('Aborting...')
                for p in process_list:
                    p.terminate()
                exit(1)
        else:
            # Single-process attacking
            attack_process(0, args, queue)
