#coding:utf-8
import os, logging
import torch
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from torch.utils.data import TensorDataset
from transformers.data.processors.utils import DataProcessor, InputExample, InputFeatures
from tqdm import tqdm


logger = logging.getLogger(__name__)


class newsProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, 'train1.txt'))

    def get_dev_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, 'dev1.txt'))

    def get_test_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, 'test1.txt'))

    def get_labels(self):
        label2id = {'娱乐': 0, '财经': 1, '体育': 2, '家居': 3, '教育': 4,
                      '房产': 5, '时尚': 6, '游戏': 7, '科技': 8, '时政': 9}
        return label2id


    def _create_examples(self, path):
        examples = []
        with open(path, mode='r', encoding='utf8') as f:
            for line in tqdm(enumerate(f.readlines())):
                id = line[0]
                line = line[1].strip()
                line = line.split('\t')
                label = line[0]
                text = line[1]
                text = ILLEGAL_CHARACTERS_RE.sub(r'', text)
                example = InputExample(guid=id, text_a=text, label=label)
                examples.append(example)
        return examples


def convert_examples_to_features(examples, tokenizer, max_length=512, label2id=None):
    logger.info("正在创建 features")
    features = []
    for (ex_index, example) in tqdm(enumerate(examples)):
        inputs = tokenizer.encode_plus(example.text_a, add_special_tokens=True, max_length=max_length,
                                       pad_to_max_length=True, truncation="longest_first")
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        attention_mask = inputs['attention_mask']
        input_len, att_mask_len, token_type_len = len(input_ids), len(attention_mask), len(token_type_ids)
        assert input_len == max_length, "input_ids 长度错误 {} vs {}".format(input_len, max_length)
        assert att_mask_len == max_length, "att_mask 长度错误 {} vs {}".format(att_mask_len, max_length)
        assert token_type_len == max_length, "token_type_ids 长度错误 {} vs {}".format(token_type_len, max_length)

        label = label2id[example.label]
        features.append(InputFeatures(input_ids=input_ids, attention_mask=attention_mask,
                                      token_type_ids=token_type_ids, label=label))

    return features


def load_and_cache_examples(args, processor, tokenizer, mode, examples=None):
    label2id = processor.get_labels()
    if mode in ['train', 'dev', 'test']:
        # features 数据保存到本地文件
        if mode == 'train':
            cached_features_file = os.path.join(args.data_dir, 'cached_train_{}'.format(str(args.max_length)))
        if mode == 'dev':
            cached_features_file = os.path.join(args.data_dir, 'cached_dev_{}'.format(str(args.max_length)))
        if mode == 'test':
            cached_features_file = os.path.join(args.data_dir, 'cached_test_{}'.format(str(args.max_length)))

        if os.path.exists(cached_features_file):
            logger.info("从本地文件加载 features，%s", cached_features_file)
            features = torch.load(cached_features_file)
        else:
            logger.info("创建本地文件保存 features，%s", args.data_dir)
            if mode == 'train':
                examples = processor.get_train_examples(args.data_dir)
            if mode == 'dev':
                examples = processor.get_dev_examples(args.data_dir)
            if mode == 'test':
                examples = processor.get_test_examples(args.data_dir)

            features = convert_examples_to_features(examples, tokenizer, args.max_length, label2id)
            logger.info("保存 features 到本地文件 %s", cached_features_file)
            torch.save(features, cached_features_file)
    else:
        features = convert_examples_to_features(examples, tokenizer, args.max_length, label2id)

    all_input_ids = torch.LongTensor([f.input_ids for f in features])
    all_attention_mask = torch.LongTensor([f.attention_mask for f in features])
    all_token_type_ids = torch.LongTensor([f.token_type_ids for f in features])
    all_labels = torch.LongTensor([f.label for f in features])
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset
