#coding:utf-8
import os
import logging
import torch
from processor import newsProcessor, load_and_cache_examples
from config import Args
from model import create_model
from train import train, evaluate, stacking


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

    args = Args().get_parser()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = newsProcessor()
    args.label2id = processor.get_labels()
    args.id2label = {i: label for i, label in enumerate(args.label2id)}
    args.output_dir = os.path.join(args.output_dir, args.bert_type)
    model, tokenizer = create_model(args)
    model.to(args.device)

    if args.do_train:
        train_dataset = load_and_cache_examples(args, processor, tokenizer, mode="train")
        train_loss = train(args, model, processor, tokenizer, train_dataset)
        logging.info("训练结束：loss {}".format(train_loss))

    if args.do_eval:
        eval_dataset = load_and_cache_examples(args, processor, tokenizer, mode="dev")
        eval = evaluate(args, model, eval_dataset)
        logging.info("验证结束：{}".format(eval))

    if args.do_stack:
        stacking(args, processor, tokenizer, model)
