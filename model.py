#coding:utf-8
import os, logging
import math
import torch
from transformers import AlbertConfig, BertTokenizer, AlbertForSequenceClassification, \
    AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from utils import ensemble_vote

logger = logging.getLogger(__name__)
MODEL_CLASSES = {
    'albert': (AlbertConfig, AlbertForSequenceClassification, BertTokenizer),
    'ernie': (AutoConfig, AutoModelForSequenceClassification, AutoTokenizer)
}


def create_model(args):
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.bert_type]
    num_labels = len(args.label2id)
    config = config_class.from_pretrained(args.model_name, num_labels=num_labels)
    model = model_class.from_pretrained(args.model_name, config=config)
    if args.baseline:
        model_file = os.path.join(args.baseline_dir, 'pytorch_model.bin')
        if os.path.exists(model_file):
            logger.info("从本地路径加载模型 {}".format(model_file))
            state_dict = torch.load(model_file)
            model.load_state_dict(state_dict, strict=False)
    else:
        # swa 保存的模型
        model_file = os.path.join(args.output_dir, 'checkpoint-100000', 'pytorch_model.bin')
        if os.path.exists(model_file):
            logger.info("从本地路径加载模型 {}".format(model_file))
            state_dict = torch.load(model_file)
            model.load_state_dict(state_dict, strict=False)

    tokenizer_kwards = {'do_lower_case': args.do_lower_case}
    tokenizer = tokenizer_class.from_pretrained(args.model_name, **tokenizer_kwards)
    return model, tokenizer


def save_model(args, model, global_step):
    output_dir = os.path.join(args.output_dir, "checkpoint-newsclss-{}".format(global_step))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_to_save = (model.module if hasattr(model, "module") else model)
    model_to_save.save_pretrained(output_dir)
    torch.save(args, os.path.join(output_dir, "training_args.conf"))
    logger.info("Saving model checkpoint to %s", output_dir)


class EnsembleModel:
    def __init__(self, model, model_path_list, device, lamb=1/3):

        self.models = []
        self.model = model
        self.lamb = lamb
        for idx, _path in enumerate(model_path_list):
            logger.info('Load model from {}'.format(_path))
            model.load_state_dict(torch.load(_path, map_location=torch.device('cpu')))
            model.to(device)
            model.eval()
            self.models.append(model)


    def weight(self, t):
        """
        牛顿冷却定律加权融合
        """
        return math.exp(-self.lamb * t)

    def predict(self, inputs):
        """
        加权融合
        """
        weight_sum = 0.
        logits = None
        attention_masks = inputs['attention_mask']

        for idx, model in enumerate(self.models):
            # 使用牛顿冷却概率融合
            weight = self.weight(idx)

            # 使用概率平均融合
            # weight = 1 / len(self.models)

            tmp_logits = model(**inputs)[0] * weight
            weight_sum += weight

            if logits is None:
                logits = tmp_logits
            else:
                logits += tmp_logits

        logits = logits / weight_sum
        _, predict = logits.max(1)
        return predict

    def vote_predict(self, inputs):
        """
        投票融合
        """
        labels_list = []
        for idx, model in enumerate(self.models):
            logits = model(**inputs)
            _, predict = logits[0].max(1)
            labels_list.append(predict.item())
        return ensemble_vote(labels_list)