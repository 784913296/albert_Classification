#coding:utf-8
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, SequentialSampler
from model import create_model, EnsembleModel
from config import Args
from processor import newsProcessor, load_and_cache_examples


lamb = 0.3
threshold = 0.9


def text_predict(examples, model, tokenizer, id2label):
    model.eval()
    labels_list = []
    for text in examples:
        text = str(text).strip()
        sentencses = ILLEGAL_CHARACTERS_RE.sub(r'', text)
        sequence_dict = tokenizer.encode_plus(sentencses, max_length=args.max_length,
                                              pad_to_max_length=True, truncation=True)
        token_ids = sequence_dict['input_ids']
        token_mask = sequence_dict['attention_mask']
        token_segment_type = tokenizer.create_token_type_ids_from_sequences(token_ids_0=token_ids[1:-1])

        token_ids = torch.LongTensor(token_ids).unsqueeze(0)
        token_mask = torch.LongTensor(token_mask).unsqueeze(0)
        token_segment_type = torch.LongTensor(token_segment_type).unsqueeze(0)
        with torch.no_grad():
            inputs = {
                'input_ids': token_ids,
                'token_type_ids': token_segment_type,
                'attention_mask': token_mask
            }
            logits = model(**inputs)

        _, predict = logits[0].max(1)
        label = id2label[predict.item()]
        labels_list.append(label)
    return labels_list


def base_predict(test_dataset, model, id2label, ensemble=False, vote=False):
    sampler = SequentialSampler(test_dataset)
    eval_dataloader = DataLoader(test_dataset, sampler=sampler, batch_size=1)

    labels_list = []
    for batch in tqdm(eval_dataloader, desc="Predicting"):
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      }

            if ensemble:
                if vote:
                    predict = model.vote_predict(inputs=inputs)
                    label = id2label[predict]
                else:
                    predict = model.predict(inputs=inputs)
                    label = id2label[predict.item()]
            else:
                logits = model(**inputs)
                _, predict = logits[0].max(1)
                label = id2label[predict.item()]

        labels_list.append(label)
    return labels_list


def single_predict(test_dataset, model, id2label):
    model.eval()
    labels = base_predict(test_dataset, model, id2label)
    return labels


# 投票/加权
def ensemble_predict(test_dataset, model, id2label, vote=True):

    # ckpt_path-ensemble.txt 模型路径列表
    with open('./ckpt_path-ensemble.txt', 'r', encoding='utf-8') as f:
        ensemble_dir_list = f.readlines()
        print('ENSEMBLE_DIR_LIST:{}'.format(ensemble_dir_list))
    model_path_list = [x.strip() for x in ensemble_dir_list]
    print('model_path_list:{}'.format(model_path_list))

    # device = torch.device(f'cuda:{GPU_IDS[0]}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EnsembleModel(model=model, model_path_list=model_path_list, device=device, lamb=lamb)
    labels = base_predict(test_dataset, model, id2label, ensemble=True, vote=True)
    return labels




if __name__ == '__main__':
    args = Args().get_parser()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = newsProcessor()
    args.label2id = processor.get_labels()
    args.id2label = {i: label for i, label in enumerate(args.label2id)}
    model, tokenizer = create_model(args)
    test_dataset = load_and_cache_examples(args, processor, tokenizer, mode='test')

    labels_list = single_predict(test_dataset, model, args.id2label)
    print(labels_list)

    labels_list = ensemble_predict(test_dataset, model, args.id2label)
    print(labels_list)

    text = ["对于我国的科技巨头华为而言，2019年注定是不平凡的一年，由于在5G领域遥遥领先于其他国家，华为遭到了不少方面的觊觎，并因此承受了太多不公平地对待，在零部件供应、核心技术研发、以及市场等多个领域受到了有意打压。但是华为并没有因此而一蹶不振，而是亮出了自己的一张又一张“底牌”，随着麒麟处理器、海思半导体以及鸿蒙操作系统的闪亮登场，华为也向世界证明了自己的实力，上演了一场几乎完美的绝地反击。"]
    label_list = text_predict(text, model, tokenizer, args.id2label)
    print(label_list)