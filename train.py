import os
import json
import random
import argparse
import torch
from tqdm import tqdm, trange
import numpy as np
from transformers import *
# from sklearn.metrics import classification_report
from active_learning.metrics import Metrics
from active_learning.utils import NerDataset, SummaryManager
from active_learning.model import BertWithCRF
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import logging
import random
import pandas as pd

'''
暂时先把device设为CPU
'''

# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')
print(f'Now the device is {device}')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def read_json(path):
    file = open(path, 'r', encoding='utf-8')
    papers = []
    for line in file.readlines():
        dic = json.loads(line)
        papers.append(dic)
    return papers


def set_seed(seed=100):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    n_gpu = torch.cuda.device_count()
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def query_func(query_list, k):
    if not query_list:
        raise ValueError
    q = sorted(query_list, key=lambda x: x[0])
    query = [i[1] for i in q[:k]]
    return query

def merge_dic(train, val):
    merge = {}
    for k, v in train.items():
        merge[k] = v
    for k, v in val.items():
        merge[k] = v
    return merge


def generate_new_file(data_dir, learn_loop, query_list, total_loops, top_min_k, mode='lc'):

    train_files = [f for f in os.listdir(data_dir) if f.startswith('training_' + str(learn_loop))]
    val_files = [f for f in os.listdir(data_dir) if f.startswith('valing_' + str(learn_loop))]
    test_files = [f for f in os.listdir(data_dir) if f.startswith('testing_' + str(learn_loop))]

    old_train = read_json(os.path.join(data_dir, train_files[0]))[0]
    old_val = read_json(os.path.join(data_dir, val_files[0]))[0]
    old_merge = merge_dic(old_train, old_val)
    #old_merge = dict(old_train, **old_val)
    old_test = read_json(os.path.join(data_dir, test_files[0]))[0]
    if mode == 'random':
        query = set()
        oti = list(old_test.items())
        while len(query) < top_min_k:
            rand = random.randint(0, len(oti))
            query.add(oti[rand][0])
        query = list(query)
    else:
        query = query_list

    tmp_dic = {}
    for q in query:
        tmp_dic[q] = old_test[q]
    new_test = {}
    for k in old_test.keys():
        if not k in query:
            new_test[k] = old_test[k]

    #new_merge = dict(old_merge, **tmp_dic)
    new_merge = merge_dic(old_merge, tmp_dic)

    train_len = int(len(new_merge) * 0.8)
    val_len = len(new_merge) - train_len
    new_train, new_val = {}, {}
    cnt = 0
    for k in new_merge:
        if k in old_train and cnt <= val_len:
            new_val[k] = new_merge[k]
            cnt += 1
        else:
            new_train[k] = new_merge[k]
    if learn_loop + 1 < total_loops:
        with open(os.path.join(data_dir, 'training_' + str(learn_loop + 1) + '.json'), 'w') as f_t:
            json.dump(new_train, f_t)
        with open(os.path.join(data_dir, 'valing_' + str(learn_loop + 1) + '.json'), 'w') as f_v:
            json.dump(new_val, f_v)
        with open(os.path.join(data_dir, 'testing_' + str(learn_loop + 1) + '.json'), 'w') as f_te:
            json.dump(new_test, f_te)
    print('Files have been splitted!')


def predict(test_dl, save_path, class_num, prefix="NER"):
    model = BertWithCRF(num_classes=class_num)
    model.load_state_dict(
        {k.replace('module.', ''): v for k, v in torch.load(save_path)["model_state_dict"].items()})
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cpu')
    model.to(device)
    model.eval()
    # Test!
    logger.info("***** Running test {} *****".format(prefix))
    list_of_y_real, list_of_pred_tags, query_list = [], [], []
    count_correct, total_count = 0, 0
    with tqdm(total=len(test_dl)) as bar:
        for idx, batch in enumerate(test_dl):
            input_index = batch["index"]
            input_ids = batch["ids"].to(device)
            seg_ids = batch["token_type_ids"].to(device)
            y_real = batch["targets"].to(device)
            with torch.no_grad():
                log_likelihood, sequence_of_tags = model(input_ids, seg_ids, y_real)
            log_likelihood = log_likelihood.data.to('cpu')
            y_real = y_real.data.to('cpu')
            sequence_of_tags = torch.tensor(sequence_of_tags).data.to('cpu')
            count_correct += (sequence_of_tags == y_real).float()[y_real != 1].sum()
            total_count += len(y_real[y_real != 1])
            for seq_elm in y_real.tolist():
                list_of_y_real += seq_elm
            for seq_elm in sequence_of_tags.tolist():
                list_of_pred_tags += seq_elm
            # a = log_likelihood.tolist()
            # b = input_index
            # print(f'log_likelihood -->{a}, input_index: {b}')
            if isinstance(input_index, list):
                for log_elm, index in zip(log_likelihood.tolist(), input_index):
                    query_list.append((log_elm, index))
            else:
                query_list.append((log_likelihood.tolist()[0], index))
            bar.update(1)
    acc = (count_correct / total_count).item() if total_count else 0.0
    result = {"test_acc": acc}
    return result, list_of_y_real, list_of_pred_tags, query_list

def train(model, tr_dl, optimizer, n_gpu, scheduler, \
          gradient_accumulation_steps, tr_loss, logging_loss, tb_writer, logging_steps, global_step, epoch):
    model.train()
    tr_summary = {}
    epoch_iterator = tqdm(tr_dl, desc="Iteration")
    for step, batch in enumerate(epoch_iterator):

        input_index = batch["index"]
        input_ids = batch["ids"].to(device)
        seg_ids = batch["token_type_ids"].to(device)
        y_real = batch["targets"].to(device)
        log_likelihood, sequence_of_tags = model(input_ids, seg_ids, y_real)
        loss = -1 * log_likelihood.mean()
        if n_gpu > 1:
            loss = loss.mean()
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
        tr_loss += loss.item()
        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1
            with torch.no_grad():
                sequence_of_tags = torch.tensor(sequence_of_tags).to(device)
                mb_acc = (sequence_of_tags == y_real).float()[y_real != 1].mean()

            tr_acc = mb_acc.item()
            tr_loss_avg = tr_loss / global_step
            tr_summary = {'loss': tr_loss_avg, 'acc': tr_acc}

            # if step % 50 == 0:
            print('epoch : {}, global_step : {}, tr_loss: {:.3f}, tr_acc: {:.2%}'.format(epoch + 1, global_step,
                                                                                         tr_summary['loss'],
                                                                                         tr_summary['acc']))
            # training & evaluation log
            if logging_steps > 0 and global_step % logging_steps == 0:
                tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                tb_writer.add_scalars('loss',
                                      {'train': (tr_loss - logging_loss) / logging_steps}, global_step)
                tb_writer.add_scalars('acc', {'train': tr_acc},
                                      global_step)

            print("Average loss: {} at global step: {}".format(
                (tr_loss - logging_loss) / logging_steps, global_step))
            logging_loss = tr_loss
    return tr_summary, tb_writer, global_step, logging_loss

def evaluate(model, val_dl, tb_writer, global_step, prefix="NER"):
    model.eval()
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cpu')
    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    eval_loss = 0.0
    nb_eval_steps = 0

    list_of_y_real = []
    list_of_pred_tags = []
    count_correct = 0
    total_count = 0

    with tqdm(total=len(val_dl)) as bar:
        for idx, batch in enumerate(val_dl):
            model.eval()
            input_index = batch["index"]
            input_ids = batch["ids"].to(device)
            seg_ids = batch["token_type_ids"].to(device)
            y_real = batch["targets"].to(device)

            with torch.no_grad():
                log_likelihood, sequence_of_tags = model(input_ids, seg_ids, y_real)
                loss = -1 * log_likelihood.mean()
                eval_loss += -1 * loss.float().item()
            nb_eval_steps += 1

            y_real = y_real.data.to('cpu')
            # print(y_real)
            sequence_of_tags = torch.tensor(sequence_of_tags).data.to('cpu')
            count_correct += (sequence_of_tags == y_real).float()[y_real != 1].sum()
            total_count += len(y_real[y_real != 1])

            for seq_elm in y_real.tolist():
                list_of_y_real += seq_elm
            # print(list_of_y_real)
            for seq_elm in sequence_of_tags.tolist():
                list_of_pred_tags += seq_elm
            # print(list_of_pred_tags)
            bar.update(1)

    eval_loss = eval_loss / nb_eval_steps
    acc = (count_correct / total_count).item() if total_count else 0.0
    result = {"eval_acc": acc, "eval_loss": eval_loss}
    tb_writer.add_scalars('loss', {'val': result["eval_loss"]}, global_step)
    return result, list_of_y_real, list_of_pred_tags, tb_writer


def save_cr_and_cm(val_ds, list_of_y_real, list_of_pred_tags, cr_save_path="classification_report.csv", prefix='Val'):
    target_names = []
    for key in val_ds.tag2id.keys():
        if key != '[PAD]' and key != 'O':
            target_names.append(key[2:])
    metr = Metrics(target_names)
    true_labels = [val_ds.id2tag[i] for i in list_of_y_real]
    pred_labels = [val_ds.id2tag[j] for j in list_of_pred_tags]
    cr_dict = metr.classification_report(pred_labels, true_labels)
    df = pd.DataFrame(cr_dict).transpose()
    df.to_csv(cr_save_path)
    logger.info("***** Running evaluation {} *****".format(prefix))
    print("*" * 20 + prefix + " Metrics " + "*" * 20)
    print(df)


def main(parser):
    # Config
    args = parser
    # args = parser.parse_args()
    # model_config = Config(json_path=model_dir / 'config.json')

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    results = {}

    # Model init
    model = BertWithCRF(num_classes=args.class_num)
    model.to(device)

    # Optim init
    train_examples_len = 1600 #2160=(200+300+400+500+600+700)*0.8
    val_examples_len = 400 #540=(200+300+400+500+600+700)*0.2
    print("num of train: {}, num of val: {}".format(train_examples_len, val_examples_len))

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    # num_train_optimization_steps = int(train_examples_len / model_config.batch_size / model_config.gradient_accumulation_steps) * model_config.epochs
    t_total = 1600 // args.gradient_accumulation_steps * args.epochs #1600 =【(200+300+400+500+600)*0.8】* epochs / batch_size

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)

    n_gpu = torch.cuda.device_count()
    print(f'gpu_num: {n_gpu}')
    # if n_gpu > 1:
    #     model = torch.nn.DataParallel(model)

    print('training')
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", 1600)  #2160=(200+300+400+500+600)*0.8
    logger.info("  Learn Loops = %d", args.total_loops)
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0

    # model.zero_grad()
    set_seed()

    #set learn loop
    learn_loop = 0
    while learn_loop < args.total_loops:
        logger.info("***** Running training at %d loop *****", learn_loop)
        # Train & Val Datasets
        train_files = [f for f in os.listdir(args.data_dir) if f.startswith('training_'+str(learn_loop))]
        val_files = [f for f in os.listdir(args.data_dir) if f.startswith('valing_'+str(learn_loop))]
        test_files = [f for f in os.listdir(args.data_dir) if f.startswith('testing_'+str(learn_loop))]

        tr_ds = NerDataset(os.path.join(args.data_dir, train_files[0]), args.max_len, tokenizer)
        val_ds = NerDataset(os.path.join(args.data_dir, val_files[0]), args.max_len, tokenizer)
        test_ds = NerDataset(os.path.join(args.data_dir, test_files[0]), args.max_len, tokenizer)

        tr_dl = DataLoader(
            tr_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=True
        )
        val_dl = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=True
        )
        test_dl = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=True
        )

        # Save init
        model_dir = os.path.join(args.root_dir, 'model_'+ args.mode + '_' + str(learn_loop))
        output_dir = os.path.join(args.root_dir, 'output_'+ args.mode + '_' + str(learn_loop))
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        tb_writer = SummaryWriter('{}/runs'.format(model_dir))
        # checkpoint_manager = CheckpointManager(args.model_dir)
        summary_manager = SummaryManager(output_dir)

        # Train!
        epoch_cnt = 0
        tr_loss, logging_loss = 0.0, 0.0
        best_dev_acc, best_dev_loss, last_dev_loss = 0.0, float("inf"), float("inf")
        best_steps, best_epoch = 0, 0
        train_iterator = trange(int(args.epochs), desc="Epoch")
        for _epoch, _ in enumerate(train_iterator):
            epoch = _epoch

            #train
            tr_summary, tb_writer, global_step, logging_loss = train(model, tr_dl, optimizer, n_gpu, scheduler, \
                  args.gradient_accumulation_steps,tr_loss, logging_loss, tb_writer, args.logging_steps, global_step, epoch)

            #save after train

            print("Saving model checkpoint to %s", model_dir)
            state = {'global_step': global_step + 1,
                     'model_state_dict': model.state_dict(),
                     'opt_state_dict': optimizer.state_dict()}
            torch.save(state, os.path.join(model_dir, 'model-trained-epoch-{}.bin'.format(epoch + 1)))
            print("Saving model checkpoint as model-trained-epoch-{}.bin".format(epoch + 1))

            #eval and save
            eval_summary, list_of_y_real, list_of_pred_tags, tb_writer = evaluate(model, val_dl, tb_writer, global_step)
            print(
                " loop: {}, epochs: {}, eval acc: {}, loss: {}, global steps: {}".format(learn_loop, epoch, eval_summary['eval_acc'],
                                                                               eval_summary['eval_loss'],
                                                                               global_step))

            summary = {'train': tr_summary, 'eval': eval_summary}
            summary_manager.update(summary)
            print("summary: ", summary)
            summary_manager.save('summary.json')
            tb_writer.add_scalars('acc', {'train': tr_summary["acc"], 'val': eval_summary["eval_acc"]}, global_step)


            is_best = eval_summary["eval_acc"] >= best_dev_acc
            if is_best:
                best_epoch = epoch + 1
                best_dev_acc = eval_summary["eval_acc"]
                best_dev_loss = eval_summary["eval_loss"]
                best_steps = global_step
                torch.save(state,os.path.join(model_dir,'best-epoch-{}-step-{}-acc-{:.3f}-better.bin'.format(epoch + 1, global_step, best_dev_acc)))
                print("Saving model checkpoint as best-epoch-{}-step-{}-acc-{:.3f}-better.bin".format(epoch + 1,
                                                                                               global_step,
                                                                                               best_dev_acc))

                # save classification report
                cr_save_path = os.path.join(output_dir,
                                            'best-epoch-{}-step-{}-acc-{:.3f}-better-cr.csv'.format(epoch + 1,
                                                                                             global_step,
                                                                                             best_dev_acc))
                save_cr_and_cm(val_ds, list_of_y_real, list_of_pred_tags, cr_save_path=cr_save_path, prefix='Val')
            else:
                torch.save(state, os.path.join(model_dir,
                                               'model-epoch-{}-step-{}-acc-{:.3f}.bin'.format(epoch + 1,
                                                                                              global_step,
                                                                                              eval_summary[
                                                                                                  "eval_acc"])))
                print("Saving model checkpoint as model-epoch-{}-step-{}-acc-{:.3f}.bin".format(epoch + 1,
                                                                                                global_step,
                                                                                                eval_summary[
                                                                                                    "eval_acc"]))
                cr_save_path = os.path.join(output_dir,
                                            'best-epoch-{}-step-{}-acc-{:.3f}-cr.csv'.format(epoch + 1,
                                                                                             global_step,
                                                                                             best_dev_acc))
                save_cr_and_cm(val_ds, list_of_y_real, list_of_pred_tags, cr_save_path=cr_save_path, prefix='Val')

            # early_stopping
            is_cnt = eval_summary["eval_loss"] >= last_dev_loss
            if is_cnt:
                epoch_cnt += 1
            else:
                epoch_cnt = 0
            if epoch_cnt >= 3:
                break

        tb_writer.close()
        print("global_step = {}, average loss = {}".format(global_step, tr_loss / global_step))
        if args.test_after_training:
            model_files = [f for f in os.listdir(model_dir) if f.startswith('best-epoch-{}'.format(best_epoch))]
            test_model_path = os.path.join(model_dir, model_files[0])
            test_summary, list_of_y_real, list_of_pred_tags, query_list = predict(test_dl, test_model_path, args.class_num)
            cr_save_path = os.path.join(model_dir, 'test_cr.csv')
            save_cr_and_cm(test_ds, list_of_y_real, list_of_pred_tags, cr_save_path=cr_save_path, prefix='Test')
            if args.active_learning:
                # start to prepare for the next loop
                print("*" * 20 + " Start to prepare for the next loop! " + "*" * 20)
                querys = query_func(query_list, args.top_min_k)
                generate_new_file(args.data_dir, learn_loop, querys, args.total_loops, args.top_min_k, mode=args.mode)
        results[str(learn_loop)] = {'global_step':global_step, 'avg_loss': tr_loss/global_step, \
                                    'best_step':best_steps, 'best_epoch':best_epoch}
        learn_loop += 1

    return results


class config:
    root_dir = './'
    data_dir = './active_data/random'
    max_len = 400
    batch_size = 2
    class_num = 20
    num_workers = 0
    epochs = 2
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    adam_epsilon = 1e-8
    warmup_steps = 1
    logging_steps = 10
    # save_steps = 482  # len(tr_dl) == len(tr_ds) // batch_size
    evaluate_during_training = True
    test_after_training = True
    active_learning = True
    total_loops = 5
    top_min_k = 100
    mode = 'random'

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--model_dir', default='../model', type=str, help="Directory to save model")
    # parser.add_argument('--data_dir', default='../', type=str, help="Directory containing .json of data")
    # parser.add_argument('--max_len', default=10, type=int, help="Max length of sequence")
    # parser.add_argument('--batch_size', default=1, type=int, help="Batch size of data")
    # parser.add_argument('--class_num', default=9, type=int, help="The number of classes of entities")
    # parser.add_argument('--num_workers', default=4, type=int, help='The number of workers when processing dataloader')
    # parser.add_argument('--epochs', default=3, help='The epochs set for training')
    # parser.add_argument('--gradient_accumulation_steps', default=1, help='gradient accumulation steps')
    # parser.add_argument('--learning_rate', default=1e-4, type=float)
    # parser.add_argument('--adam_epsilon', default=1e-8, type=float)
    # parser.add_argument('--warmup_steps', default=3, type=int)
    # parser.add_argument('--logging_steps', default=1, type=int， help="")
    # parser.add_argument('--save_steps', default=1, type=int, help="The save steps are suppose to be the same as  the len(dataloader).)
    # parser.add_argument('--evaluate_during_training', action='store_true', default=False)
    parser = config()
    # res = main(parser)
    # print("All finished!!")
    print("Start to test!!!")
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    test_ds = NerDataset(os.path.join('./active_data/random/', 'final_test.json'), 400, tokenizer)
    test_dl = DataLoader(
        test_ds,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )
    model_dir = "C:/Users/yang guangyu/Desktop/医疗NER/active_learning/random/model_random_4/"
    model_name = "best-epoch-2-step-1592-acc-0.984-better.bin"
    test_model_path = os.path.join(model_dir, model_name)
    # model_files = [f for f in os.listdir(model_dir) if f.startswith('best-epoch-{}'.format(best_epoch))]
    # test_model_path = os.path.join(model_dir, model_files[0])
    test_summary, list_of_y_real, list_of_pred_tags, query_list = predict(test_dl, test_model_path, 20)
    cr_save_path = os.path.join(model_dir, 'final_test_cr_random.csv')
    save_cr_and_cm(test_ds, list_of_y_real, list_of_pred_tags, cr_save_path=cr_save_path, prefix='Test')




