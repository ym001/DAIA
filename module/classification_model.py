#!/usr/bin/env python
# coding: utf-8


from __future__ import absolute_import, division, print_function

import os
import math
import json
import random
import warnings

from multiprocessing import cpu_count

import torch
import numpy as np
import gc
from scipy.stats import pearsonr, mode
from sklearn.metrics import mean_squared_error, matthews_corrcoef, confusion_matrix, label_ranking_average_precision_score,accuracy_score
from tensorboardX import SummaryWriter
from tqdm.auto import trange, tqdm

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
    TensorDataset
)

from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import (
    WEIGHTS_NAME,
    BertConfig,BertTokenizer,
    XLNetConfig,XLNetTokenizer,
    XLMConfig, XLMTokenizer,
    RobertaConfig, RobertaTokenizer,
    DistilBertConfig, DistilBertTokenizer,
    AlbertConfig, AlbertTokenizer,
    CamembertConfig, CamembertTokenizer
)

from classification_utils import (
    InputExample,
    convert_examples_to_features
)
import sys
####################
#importation des modules perso
path = "../models"
sys.path.append(path)

from bert_model import BertForSequenceClassification
from roberta_model import RobertaForSequenceClassification
from xlm_model import XLMForSequenceClassification
from xlnet_model import XLNetForSequenceClassification
from distilbert_model import DistilBertForSequenceClassification
from albert_model import AlbertForSequenceClassification
from camembert_model import CamembertForSequenceClassification


class ClassificationModel:
    def __init__(self, model_type, model_name, num_labels=None, weight=None, args=None, use_cuda=True, cuda_device=-1,early_stopping=False):
		
        """
        Initializes a ClassificationModel model.

        Args:
            model_type: The type of model (bert, xlnet, xlm, roberta, distilbert)
            model_name: Default Transformer model name or path to a directory containing Transformer model file (pytorch_nodel.bin).
            num_labels (optional): The number of labels or classes in the dataset.
            weight (optional): A list of length num_labels containing the weights to assign to each label for loss calculation.
            args (optional): Default args will be used if this parameter is not provided. If provided, it should be a dict containing the args that should be changed in the default args.
            use_cuda (optional): Use GPU if available. Setting to False will force model to use CPU only.
            cuda_device (optional): Specific GPU that should be used. Will use the first available GPU by default.
        """
        torch.manual_seed(6)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        MODEL_CLASSES = {
            'bert':       (BertConfig, BertForSequenceClassification, BertTokenizer),
            'scibert':       (BertConfig, BertForSequenceClassification, BertTokenizer),
            'clinicalbert':       (BertConfig, BertForSequenceClassification, BertTokenizer),
            'xlnet':      (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
            'xlm':        (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
            'roberta':    (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
            'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
            'albert':     (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
            'camembert':  (CamembertConfig, CamembertForSequenceClassification, CamembertTokenizer)
        }
        self.early_stopping=early_stopping
        self.config_class, self.model_class, tokenizer_class = MODEL_CLASSES[model_type]
        
        if num_labels:
            self.config = self.config_class.from_pretrained(model_name, num_labels=num_labels,output_hidden_states=True)
            self.num_labels = num_labels
        else:
            self.config = self.config_class.from_pretrained(model_name,output_hidden_states=True)
            self.num_labels = self.config.num_labels
        self.weight = weight

        if use_cuda:
            if torch.cuda.is_available():
                if cuda_device == -1:
                    self.device = torch.device("cuda")
                else:
                    self.device = torch.device(f"cuda:{cuda_device}")
            else:
                raise ValueError("'use_cuda' set to True when cuda is unavailable. Make sure CUDA is available or set use_cuda=False.")
        else:
            self.device = "cpu"

        if self.weight:
            self.model = self.model_class.from_pretrained(model_name, config=self.config, weight=torch.Tensor(self.weight).to(self.device))
        else:
            self.model = self.model_class.from_pretrained(model_name, config=self.config)

        self.results = {}
        self.max_seq_length=128
        self.args = {
            'output_dir': 'outputs/',
            'cache_dir': 'cache_dir/',

            'fp16': True,
            'fp16_opt_level': 'O1',
            'max_seq_length': self.max_seq_length,
            'train_batch_size': 8,
            'gradient_accumulation_steps': 1,
            'eval_batch_size': 8,
            'num_train_epochs': 1,
            'weight_decay': 0,
            'learning_rate': 4e-5,
            'adam_epsilon': 1e-8,
            'warmup_ratio': 0.06,
            'warmup_steps': 0,
            'max_grad_norm': 1.0,
            'do_lower_case': False,

            'logging_steps': 50,
            'save_steps': 2000,
            'evaluate_during_training': False,
            'evaluate_during_training_steps': 2000,

            'overwrite_output_dir': False,
            'reprocess_input_data': False,

            'process_count': cpu_count() - 2 if cpu_count() > 2 else 1,
            'n_gpu': 1,
            'use_multiprocessing': True,
            'silent': False,

            'sliding_window': False,
            'tie_value': 1,
            'stride': 0.8
        }

        if not use_cuda:
            self.args['fp16'] = False

        if args:
            self.args.update(args)

        self.tokenizer = tokenizer_class.from_pretrained(model_name, do_lower_case=self.args['do_lower_case'])

        self.args['model_name'] = model_name
        self.args['model_type'] = model_type

        if model_type == 'camembert':
            warnings.warn("use_multiprocessing automatically disabled as CamemBERT fails when using multiprocessing for feature conversion.")
            self.args['use_multiprocessing'] = False

    def train_model(self, train_df, multi_label=False, output_dir=None, show_running_loss=True, args=None, eval_df=None, **kwargs):
        """
        Trains the model using 'train_df'

        Args:
            train_df: Pandas Dataframe containing at least two columns. If the Dataframe has a header, it should contain a 'text' and a 'labels' column. If no header is present,
            the Dataframe should contain at least two columns, with the first column containing the text, and the second column containing the label. The model will be trained on this Dataframe.
            output_dir: The directory where model files will be saved. If not given, self.args['output_dir'] will be used.
            show_running_loss (optional): Set to False to prevent running loss from being printed to console. Defaults to True.
            args (optional): Optional changes to the args dict of the model. Any changes made will persist for the model.
            eval_df (optional): A DataFrame against which evaluation will be performed when evaluate_during_training is enabled. Is required if evaluate_during_training is enabled.
            **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use). E.g. f1=sklearn.metrics.f1_score.
                        A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions.

        Returns:
            None
        """

        if args:
            self.args.update(args)

        if self.args['silent']:
            show_running_loss = False

        if self.args['evaluate_during_training'] and eval_df is None:
            raise ValueError("evaluate_during_training is enabled but eval_df is not specified. Pass eval_df to model.train_model() if using evaluate_during_training.")

        if not output_dir:
            output_dir = self.args['output_dir']

        if os.path.exists(output_dir) and os.listdir(output_dir) and not self.args["overwrite_output_dir"]:
            raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(output_dir))

        self._move_model_to_device()

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        global_step, tr_loss = self.train(train_df, output_dir, show_running_loss=show_running_loss, eval_df=eval_df, **kwargs)

    
    def train(self, train_dataset, output_dir, show_running_loss=True, eval_df=None, **kwargs):
        """
        Trains the model on train_dataset.

        Utility function to be used by the train_model() method. Not intended to be used directly.
        """

        args = self.args

        tb_writer = SummaryWriter()
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args["train_batch_size"])

        t_total = len(train_dataloader) // args["gradient_accumulation_steps"] * args["num_train_epochs"]

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.model.named_parameters() if not any(
                nd in n for nd in no_decay)], "weight_decay": args["weight_decay"]},
            {"params": [p for n, p in self.model.named_parameters() if any(
                nd in n for nd in no_decay)], "weight_decay": 0.0}
        ]

        warmup_steps = math.ceil(t_total * args["warmup_ratio"])
        args["warmup_steps"] = warmup_steps if args["warmup_steps"] == 0 else args["warmup_steps"]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args["learning_rate"], eps=args["adam_epsilon"])
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args["warmup_steps"], num_training_steps=t_total)

        if args["fp16"]:
            try:
                from apex import amp
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

            self.model, optimizer = amp.initialize(self.model, optimizer, opt_level=args["fp16_opt_level"])

        if args["n_gpu"] > 1:
            self.model = torch.nn.DataParallel(self.model)

        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        self.model.zero_grad()
        train_iterator = trange(int(args["num_train_epochs"]), desc="Epoch", disable=args['silent'])
        epoch_number = 0

        self.model.train()
        for _ in train_iterator:
            for step, batch in enumerate(tqdm(train_dataloader, desc="Current iteration", disable=args['silent'])):
                batch = tuple(t.to(self.device) for t in batch)

                inputs = self._get_inputs_dict(batch)
                outputs = self.model(**inputs)
                # model outputs are always tuple in pytorch-transformers (see doc)
                loss = outputs[0]

                if args['n_gpu'] > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                    
                if show_running_loss:
                    print("\rRunning loss: %f" % loss, end="")

                if args["gradient_accumulation_steps"] > 1:
                    loss = loss / args["gradient_accumulation_steps"]

                if args["fp16"]:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args["max_grad_norm"])
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args["max_grad_norm"])

                tr_loss += loss.item()
                if (step + 1) % args["gradient_accumulation_steps"] == 0:
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

                    if args["logging_steps"] > 0 and global_step % args["logging_steps"] == 0:
                        # Log metrics
                        tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                        tb_writer.add_scalar("loss", (tr_loss - logging_loss)/args["logging_steps"], global_step)
                        logging_loss = tr_loss

            if self.early_stopping:
                predictions_validation,emb=self.predict(self.x_validation)
                label_predits_validation=np.argmax(predictions_validation, axis=1)
                acc_validation=accuracy_score(self.y_validation,label_predits_validation)
                es(acc_validation, self.model)
                 
                if es.early_stop:
                    print("Early stopping")
                    break
                    
            epoch_number += 1
            output_dir_current = os.path.join(output_dir, "epoch-{}".format(epoch_number))

        del train_sampler,train_dataloader
        torch.cuda.empty_cache()

        return global_step, tr_loss / global_step
        
    def extraction_features(self,train_df):
        self._move_model_to_device()
        train_examples = [InputExample(i, text, None, label) for i, (text, label) in enumerate(zip(train_df['text'], train_df['labels']))]
        return self.load_and_cache_examples(train_examples)

    def load_and_cache_examples(self, examples, evaluate=False, no_cache=True, multi_label=False):
        """
        Converts a list of InputExample objects to a TensorDataset containing InputFeatures. Caches the InputFeatures.

        Utility function for train() and eval() methods. Not intended to be used directly.
        """

        process_count = self.args["process_count"]

        output_mode = "classification"
        args = self.args

        if not os.path.isdir(self.args["cache_dir"]):
            os.mkdir(self.args["cache_dir"])

        mode = "dev" if evaluate else "train"
        cached_features_file = os.path.join(args["cache_dir"], "cached_{}_{}_{}_{}_{}".format(mode, args["model_type"], args["max_seq_length"], self.num_labels, len(examples)))

        if os.path.exists(cached_features_file) and not args["reprocess_input_data"] and not no_cache:
            features = torch.load(cached_features_file)
            print(f"Features loaded from cache at {cached_features_file}")
        else:
            print(f"Converting to features started.")
            features = convert_examples_to_features(
                examples,
                args["max_seq_length"],
                self.tokenizer,
                output_mode,
                # XLNet has a CLS token at the end
                cls_token_at_end=bool(args["model_type"] in ["xlnet"]),
                cls_token=self.tokenizer.cls_token,
                cls_token_segment_id=2 if args["model_type"] in ["xlnet"] else 0,
                sep_token=self.tokenizer.sep_token,
                # RoBERTa uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                sep_token_extra=bool(args["model_type"] in ["roberta"]),
                # PAD on the left for XLNet
                pad_on_left=bool(args["model_type"] in ["xlnet"]),
                pad_token=self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0],
                pad_token_segment_id=4 if args["model_type"] in ["xlnet"] else 0,
                process_count=process_count,
                multi_label=multi_label,
                silent=args['silent'],
                use_multiprocessing=args['use_multiprocessing'],
                sliding_window=args['sliding_window'],
                flatten=not evaluate,
                stride=args['stride']
            )

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

        return TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    def extraction_features_ym(self,train_df):
        self._move_model_to_device()
        train_examples = [InputExample(i, text, None, label) for i, (text, label) in enumerate(zip(train_df['text'], train_df['labels']))]
        return self.load_and_cache_examples_ym(train_examples)

    def load_and_cache_examples_ym(self, examples, evaluate=False, no_cache=True, multi_label=False):
        """
        Converts a list of InputExample objects to a TensorDataset containing InputFeatures. Caches the InputFeatures.

        Utility function for train() and eval() methods. Not intended to be used directly.
        """

        process_count = self.args["process_count"]
        output_mode = "classification"
        args = self.args

        if not os.path.isdir(self.args["cache_dir"]):
            os.mkdir(self.args["cache_dir"])

        mode = "dev" if evaluate else "train"
        cached_features_file = os.path.join(args["cache_dir"], "cached_{}_{}_{}_{}_{}".format(mode, args["model_type"], args["max_seq_length"], self.num_labels, len(examples)))

        if os.path.exists(cached_features_file) and not args["reprocess_input_data"] and not no_cache:
            features = torch.load(cached_features_file)
            print(f"Features loaded from cache at {cached_features_file}")
        else:
            print(f"Converting to features started.")
            features = convert_examples_to_features(
                examples,
                args["max_seq_length"],
                self.tokenizer,
                output_mode,
                # XLNet has a CLS token at the end
                cls_token_at_end=bool(args["model_type"] in ["xlnet"]),
                cls_token=self.tokenizer.cls_token,
                cls_token_segment_id=2 if args["model_type"] in ["xlnet"] else 0,
                sep_token=self.tokenizer.sep_token,
                # RoBERTa uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                sep_token_extra=bool(args["model_type"] in ["roberta"]),
                # PAD on the left for XLNet
                pad_on_left=bool(args["model_type"] in ["xlnet"]),
                pad_token=self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0],
                pad_token_segment_id=4 if args["model_type"] in ["xlnet"] else 0,
                process_count=process_count,
                multi_label=multi_label,
                silent=args['silent'],
                use_multiprocessing=args['use_multiprocessing'],
                sliding_window=args['sliding_window'],
                flatten=not evaluate,
                stride=args['stride']
            )


        return features

    def to_dataset_ym(self, features):
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        return TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    def predict(self, to_predict):
        """
        Performs predictions on a list of text.

        Args:
            to_predict: A python list of text (str) to be sent to the model for prediction.

        Returns:
            preds: A python list of the predictions (0 or 1) for each text.
            model_outputs: A python list of the raw model outputs for each text.
        """

        self._move_model_to_device()

        eval_sampler = SequentialSampler(to_predict)
        eval_dataloader = DataLoader(to_predict, sampler=eval_sampler, batch_size=self.args["eval_batch_size"])

        preds = None
        embeddings = None
        
        for batch in tqdm(eval_dataloader, disable=self.args['silent']):
            with torch.no_grad():
                batch = tuple(t.to(self.device) for t in batch)

                inputs = self._get_inputs_dict(batch)
                outputs = self.model(**inputs)
                tmp_eval_loss, logits ,hidden_states_tuple= outputs[:3]
                logits=torch.softmax(logits, dim=1)

            if preds is None:
                preds = logits.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            if embeddings is None:
                embeddings = hidden_states_tuple[0].detach().cpu().numpy()
            else:
                embeddings = np.append(embeddings, hidden_states_tuple[0].detach().cpu().numpy(), axis=0)
                
        return preds,embeddings


    def _threshold(self, x, threshold):
        if x >= threshold:
            return 1
        return 0


    def _move_model_to_device(self):
        self.model.to(self.device)


    def _get_inputs_dict(self, batch):
        inputs = {
            "input_ids":      batch[0],
            "attention_mask": batch[1],
            "labels":         batch[3]
        }

        # XLM, DistilBERT and RoBERTa don't use segment_ids
        if self.args["model_type"] != "distilbert":
            inputs["token_type_ids"] = batch[2] if self.args["model_type"] in ["bert", "xlnet"] else None
        return inputs
