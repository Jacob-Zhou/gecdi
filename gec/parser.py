# -*- coding: utf-8 -*-

from functools import partial
import json
import os
from datetime import datetime, timedelta
import shutil
import tempfile
from typing import Iterable, Union

import math
import dill
import torch
import torch.distributed as dist
from gec.data import Dataset
from gec.fn import map_token_ids
from supar.parser import Parser
from supar.utils import Config
from supar.utils.common import MIN, NUL, UNK
from supar.utils.field import RawField
from supar.utils.fn import set_rng_state
from supar.utils.logging import get_logger, init_logger, progress_bar
from supar.utils.metric import Metric
from supar.utils.optim import PolynomialLR
from supar.utils.parallel import DistributedDataParallel as DDP, gather, is_dist
from supar.utils.parallel import is_master
from supar.utils.tokenizer import TransformerTokenizer
from supar.utils.transform import AttachJuxtaposeTree, Batch
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn.functional import embedding

from .metric import PerplexityMetric, SpanMetric
from .model import Seq2SeqDetectModel, Seq2SeqModel
from .transform import Field, Text, Tree

logger = get_logger(__name__)


class Seq2SeqParser(Parser):

    NAME = 'seq2seq'
    MODEL = Seq2SeqModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.SRC = self.transform.SRC
        self.TGT = self.transform.TGT

    def train(self,
              train: Union[str, Iterable],
              dev: Union[str, Iterable],
              test: Union[str, Iterable],
              epochs: int,
              patience: int,
              batch_size: int = 5000,
              update_steps: int = 1,
              buckets: int = 32,
              workers: int = 0,
              clip: float = 5.0,
              amp: bool = False,
              cache: bool = False,
              verbose: bool = True,
              **kwargs) -> None:
        args = self.args.update(locals())
        init_logger(logger, verbose=args.verbose)

        self.transform.train()
        batch_size = batch_size // update_steps
        if dist.is_initialized():
            batch_size = batch_size // dist.get_world_size()
        logger.info("Loading the data")
        if args.cache:
            args.bin = os.path.join(os.path.dirname(args.path), 'bin')
        train = Dataset(self.transform, args.train,
                        **args).build(batch_size,
                                      buckets,
                                      True,
                                      dist.is_initialized(),
                                      workers,
                                      chunk_size=args.chunk_size,
                                      seed=args.seed)
        dev = Dataset(self.transform, args.dev,
                      **args).build(batch_size, buckets, False,
                                    dist.is_initialized(), workers)
        logger.info(f"{'train:':6} {train}")
        if not args.test:
            logger.info(f"{'dev:':6} {dev}\n")
        else:
            test = Dataset(self.transform, args.test,
                           **args).build(batch_size, buckets, False,
                                         dist.is_initialized(), workers)
            logger.info(f"{'dev:':6} {dev}")
            logger.info(f"{'test:':6} {test}\n")

        self.optimizer = AdamW(self.model.parameters(), args.lr,
                               (args.mu, args.nu), args.eps, args.weight_decay)
        steps = len(train.loader) * epochs // args.update_steps
        self.scheduler = PolynomialLR(self.optimizer,
                                        warmup_steps=self.args.warmup_steps,
                                        steps=steps)
        self.scaler = GradScaler(enabled=args.amp)

        if dist.is_initialized():
            self.model = DDP(self.model,
                             device_ids=[args.local_rank],
                             find_unused_parameters=args.get(
                                 'find_unused_parameters', True))
            if args.amp:
                from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook
                self.model.register_comm_hook(dist.group.WORLD,
                                              fp16_compress_hook)

        self.step, self.epoch, self.best_e, self.patience, self.n_batches = 1, 1, 1, patience, len(
            train.loader)
        self.best_metric, self.elapsed = Metric(), timedelta()
        if self.args.checkpoint:
            try:
                self.optimizer.load_state_dict(
                    self.checkpoint_state_dict.pop('optimizer_state_dict'))
                self.scheduler.load_state_dict(
                    self.checkpoint_state_dict.pop('scheduler_state_dict'))
                self.scaler.load_state_dict(
                    self.checkpoint_state_dict.pop('scaler_state_dict'))
                set_rng_state(self.checkpoint_state_dict.pop('rng_state'))
                for k, v in self.checkpoint_state_dict.items():
                    setattr(self, k, v)
                train.loader.batch_sampler.epoch = self.epoch
            except AttributeError:
                logger.warning(
                    "No checkpoint found. Try re-launching the traing procedure instead"
                )

        for epoch in range(self.epoch, args.epochs + 1):
            start = datetime.now()
            bar, metric = progress_bar(train.loader), Metric()

            logger.info(f"Epoch {epoch} / {args.epochs}:")
            self.model.train()
            if self.epoch == 1:
                torch.cuda.empty_cache()
            with self.join():
                # we should zero `step` as the number of batches in different processes is not necessarily equal
                self.step = 0
                for batch in bar:
                    with self.sync():
                        with torch.autocast(self.device,
                                            enabled=self.args.amp):
                            loss = self.train_step(batch)
                        self.backward(loss)
                    if self.sync_grad:
                        self.clip_grad_norm_(self.model.parameters(),
                                             self.args.clip)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.scheduler.step()
                        self.optimizer.zero_grad(True)
                    bar.set_postfix_str(
                        f"lr: {self.scheduler.get_last_lr()[0]:.4e} - loss: {loss:.4f}"
                    )
                    self.step += 1
                logger.info(f"{bar.postfix}")
            self.model.eval()
            with self.join(), torch.autocast(self.device,
                                             enabled=self.args.amp):
                metric = self.reduce(
                    sum([self.eval_step(i) for i in progress_bar(dev.loader)],
                        Metric()))
                logger.info(f"{'dev:':5} {metric}")
                if args.test:
                    test_metric = sum(
                        [self.eval_step(i) for i in progress_bar(test.loader)],
                        Metric())
                    logger.info(f"{'test:':5} {self.reduce(test_metric)}")

            t = datetime.now() - start
            self.epoch += 1
            self.patience -= 1
            self.elapsed += t

            if metric > self.best_metric:
                self.best_e, self.patience, self.best_metric = epoch, patience, metric
                if is_master():
                    self.save_checkpoint(args.path)
                logger.info(f"{t}s elapsed (saved)\n")
            else:
                logger.info(f"{t}s elapsed\n")
            if self.patience < 1:
                break
        if dist.is_initialized():
            dist.barrier()

        best = self.load(**args)
        # only allow the master device to save models
        if is_master():
            best.save(args.path)

        logger.info(f"Epoch {self.best_e} saved")
        logger.info(f"{'dev:':5} {self.best_metric}")
        if args.test:
            best.model.eval()
            with best.join():
                test_metric = sum(
                    [best.eval_step(i) for i in progress_bar(test.loader)],
                    Metric())
                logger.info(f"{'test:':5} {best.reduce(test_metric)}")
        logger.info(f"{self.elapsed}s elapsed, {self.elapsed / epoch}s/epoch")

    def evaluate(self,
                 data: Union[str, Iterable],
                 batch_size: int = 5000,
                 buckets: int = 8,
                 workers: int = 0,
                 amp: bool = False,
                 cache: bool = False,
                 punct: bool = False,
                 tree: bool = True,
                 proj: bool = False,
                 partial: bool = False,
                 verbose: bool = True,
                 **kwargs):
        return super().evaluate(**Config().update(locals()))

    def predict(self,
                data: Union[str, Iterable],
                pred: str = None,
                lang: str = None,
                prob: bool = False,
                batch_size: int = 5000,
                buckets: int = 8,
                workers: int = 0,
                amp: bool = False,
                cache: bool = False,
                tree: bool = True,
                proj: bool = False,
                verbose: bool = True,
                **kwargs):
        return super().predict(**Config().update(locals()))

    def train_step(self, batch: Batch) -> torch.Tensor:
        src, tgt = batch
        src_mask, tgt_mask = batch.mask, tgt.ne(self.args.pad_index)
        x = self.model(src)
        loss = self.model.loss(x, tgt, src_mask, tgt_mask)
        return loss

    @torch.no_grad()
    def eval_step(self, batch: Batch) -> PerplexityMetric:
        src, tgt = batch
        src_mask, tgt_mask = batch.mask, tgt.ne(self.args.pad_index)
        x = self.model(src)
        loss = self.model.loss(x, tgt, src_mask, tgt_mask)
        preds = golds = None
        if self.args.eval_tgt:
            golds = [(s.values[0], s.values[1]) for s in batch.sentences]
            preds = [(s.values[0], self.TGT.tokenize.decode(i[0]))
                     for s, i in zip(batch.sentences,
                                     self.model.decode(x, batch.mask).tolist())
                     ]
        return PerplexityMetric(loss, preds, golds, tgt_mask,
                                not self.args.eval_tgt)

    @torch.no_grad()
    def pred_step(self, batch: Batch) -> Batch:
        src, = batch
        x = self.model(src)
        tgt = self.model.decode(x, batch.mask)
        batch.tgt = [[self.TGT.tokenize.decode(cand) for cand in i]
                     for i in tgt.tolist()]
        return batch

    @classmethod
    def build(cls, path, min_freq=2, fix_len=20, **kwargs):
        r"""
        Build a brand-new Parser, including initialization of all data fields and model parameters.

        Args:
            path (str):
                The path of the model to be saved.
            min_freq (str):
                The minimum frequency needed to include a token in the vocabulary. Default: 2.
            fix_len (int):
                The max length of all subword pieces. The excess part of each piece will be truncated.
                Required if using CharLSTM/BERT.
                Default: 20.
            kwargs (dict):
                A dict holding the unconsumed arguments.
        """

        args = Config(**locals())
        os.makedirs(os.path.dirname(path) or './', exist_ok=True)
        if os.path.exists(path) and not args.build:
            return cls.load(**args)

        logger.info("Building the fields")
        t = TransformerTokenizer(name=args.bart)
        SRC = Field('src',
                    pad=t.pad,
                    unk=t.unk,
                    bos=t.bos,
                    eos=t.eos,
                    tokenize=t)
        TGT = Field('tgt',
                    pad=t.pad,
                    unk=t.unk,
                    bos=t.bos,
                    eos=t.eos,
                    tokenize=t)
        transform = Text(SRC=SRC, TGT=TGT)

        # share the vocab
        SRC.vocab = TGT.vocab = t.vocab
        args.update({
            'n_words': len(SRC.vocab),
            'pad_index': SRC.pad_index,
            'unk_index': SRC.unk_index,
            'bos_index': SRC.bos_index,
            'eos_index': SRC.eos_index
        })
        logger.info(f"{transform}")
        logger.info("Building the model")
        model = cls.MODEL(**args)
        logger.info(f"{model}\n")

        parser = cls(args, model, transform)
        parser.model.to(parser.device)
        return parser


class Seq2SeqDetector(Seq2SeqParser):

    NAME = 'seq2seq'
    MODEL = Seq2SeqDetectModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.SRC = self.transform.SRC
        self.TGT = self.transform.TGT
        (_, self.TGT_ERROR) = self.transform.TGTERROR

    def train(self,
              train: Union[str, Iterable],
              dev: Union[str, Iterable],
              test: Union[str, Iterable],
              epochs: int,
              patience: int,
              batch_size: int = 5000,
              update_steps: int = 1,
              buckets: int = 32,
              workers: int = 0,
              clip: float = 5.0,
              amp: bool = False,
              cache: bool = False,
              verbose: bool = True,
              **kwargs) -> None:
        args = self.args.update(locals())
        init_logger(logger, verbose=args.verbose)

        self.transform.train()
        batch_size = batch_size // update_steps
        if dist.is_initialized():
            batch_size = batch_size // dist.get_world_size()
        logger.info("Loading the data")
        if args.cache:
            if args.bin_path is None:
                args.bin = os.path.join(os.path.dirname(args.path), 'bin')
            else:
                args.bin = args.bin_path
        train = Dataset(self.transform, args.train,
                        **args).build(batch_size,
                                      buckets,
                                      True,
                                      dist.is_initialized(),
                                      workers,
                                      chunk_size=args.chunk_size,
                                      seed=args.seed)
        dev = Dataset(self.transform, args.dev,
                      **args).build(batch_size, buckets, False,
                                    dist.is_initialized(), workers)
        logger.info(f"{'train:':6} {train}")
        if not args.test:
            logger.info(f"{'dev:':6} {dev}\n")
        else:
            test = Dataset(self.transform, args.test,
                           **args).build(batch_size, buckets, False,
                                         dist.is_initialized(), workers)
            logger.info(f"{'dev:':6} {dev}")
            logger.info(f"{'test:':6} {test}\n")

        def ged_param(name):
            if name.startswith("encoder."):
                return False
            elif name.startswith("decoder."):
                return False
            else:
                return True

        no_decay = []
        self.optimizer = AdamW([{
            'params':
            p,
            'lr':
            args.lr * (1 if not ged_param(n) else args.lr_rate),
            "weight_decay":
            args.weight_decay if not any(nd in n for nd in no_decay) else 0.0,
        } for n, p in self.model.named_parameters()], args.lr,
            (args.mu, args.nu), args.eps, args.weight_decay)
        self.scheduler = ExponentialLR(self.optimizer,
                                       args.decay**(1 / args.decay_steps))

        self.scaler = GradScaler(enabled=args.amp)

        if dist.is_initialized():
            self.model = DDP(self.model,
                             device_ids=[args.local_rank],
                             find_unused_parameters=args.get(
                                 'find_unused_parameters', True))
            if args.amp:
                from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook
                self.model.register_comm_hook(dist.group.WORLD,
                                              fp16_compress_hook)

        self.step, self.epoch, self.best_e, self.patience, self.n_batches = 1, 1, 1, patience, len(
            train.loader)
        self.best_metric, self.elapsed = Metric(), timedelta()
        if self.args.checkpoint:
            try:
                self.optimizer.load_state_dict(
                    self.checkpoint_state_dict.pop('optimizer_state_dict'))
                self.scheduler.load_state_dict(
                    self.checkpoint_state_dict.pop('scheduler_state_dict'))
                self.scaler.load_state_dict(
                    self.checkpoint_state_dict.pop('scaler_state_dict'))
                set_rng_state(self.checkpoint_state_dict.pop('rng_state'))
                for k, v in self.checkpoint_state_dict.items():
                    setattr(self, k, v)
                train.loader.batch_sampler.epoch = self.epoch
            except AttributeError:
                logger.warning(
                    "No checkpoint found. Try re-launching the traing procedure instead"
                )

        for epoch in range(self.epoch, args.epochs + 1):
            start = datetime.now()
            bar, metric = progress_bar(train.loader), Metric()

            logger.info(f"Epoch {epoch} / {args.epochs}:")
            self.model.train()
            if self.epoch == 1:
                torch.cuda.empty_cache()
            with self.join():
                # we should zero `step` as the number of batches in different processes is not necessarily equal
                self.step = 0
                for batch in bar:
                    with self.sync():
                        with torch.autocast(self.device,
                                            enabled=self.args.amp):
                            loss = self.train_step(batch)
                        self.backward(loss)
                    if self.sync_grad:
                        self.clip_grad_norm_(self.model.parameters(),
                                             self.args.clip)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.scheduler.step()
                        self.optimizer.zero_grad(True)
                    bar.set_postfix_str(
                        f"lr: {self.scheduler.get_last_lr()[0]:.4e} - loss: {loss:.4f}"
                    )
                    self.step += 1
                logger.info(f"{bar.postfix}")
            self.model.eval()
            with self.join(), torch.autocast(self.device,
                                             enabled=self.args.amp):
                metric = self.reduce(
                    sum([self.eval_step(i) for i in progress_bar(dev.loader)],
                        Metric()))
                logger.info(f"{'dev:':5} {metric}")
                if args.test:
                    test_metric = sum(
                        [self.eval_step(i) for i in progress_bar(test.loader)],
                        Metric())
                    logger.info(f"{'test:':5} {self.reduce(test_metric)}")

            t = datetime.now() - start
            self.epoch += 1
            self.patience -= 1
            self.elapsed += t

            if metric > self.best_metric:
                self.best_e, self.patience, self.best_metric = epoch, patience, metric
                if is_master():
                    self.save_checkpoint(args.path)
                logger.info(f"{t}s elapsed (saved)\n")
            else:
                logger.info(f"{t}s elapsed\n")
            if self.patience < 1:
                break
        if dist.is_initialized():
            dist.barrier()

        best = self.load(**args)
        # only allow the master device to save models
        if is_master():
            best.save(args.path)

        logger.info(f"Epoch {self.best_e} saved")
        logger.info(f"{'dev:':5} {self.best_metric}")
        if args.test:
            best.model.eval()
            with best.join():
                test_metric = sum(
                    [best.eval_step(i) for i in progress_bar(test.loader)],
                    Metric())
                logger.info(f"{'test:':5} {best.reduce(test_metric)}")
        logger.info(f"{self.elapsed}s elapsed, {self.elapsed / epoch}s/epoch")

    def train_step(self, batch: Batch) -> torch.Tensor:
        src, tgt, _, src_error, _, tgt_error = batch
        src_mask, tgt_mask = src.ne(self.args.pad_index), tgt.ne(
            self.args.pad_index)
        x = self.model(src)
        loss = self.model.loss(x, tgt, src_error, tgt_error, src_mask,
                               tgt_mask)
        return loss

    @torch.no_grad()
    def eval_step(self, batch: Batch) -> PerplexityMetric:
        src, tgt, _, src_error, tgt_error_raw, tgt_error = batch
        src_mask, tgt_mask = src.ne(self.args.pad_index), tgt.ne(
            self.args.pad_index)
        x = self.model(src)
        loss = self.model.loss(x, tgt, src_error, tgt_error, src_mask,
                               tgt_mask)

        ged_golds = [self.error_label_factorize(e) for e in tgt_error_raw]
        ged_preds = [
            self.error_label_factorize(
                [self.TGT_ERROR.vocab[i] for i in e if i >= 0])
            for e in self.model.decode(x, tgt, src_mask, tgt_mask).tolist()
        ]

        return SpanMetric(loss, ged_preds, ged_golds)

    @torch.no_grad()
    def pred_step(self, batch: Batch) -> Batch:
        # src, = batch
        src, tgt = batch
        src_mask, tgt_mask = src.ne(self.args.pad_index), tgt.ne(
            self.args.pad_index)
        x = self.model(src)

        ged_preds = [
            self.error_label_factorize(
                [self.TGT_ERROR.vocab[i] for i in e if i >= 0])
            for e in self.model.decode(x, tgt, src_mask, tgt_mask).tolist()
        ]


        def json_repr(sentence, tgt_subword, pred):
            return json.dumps({
                "src_text": sentence.values[0],
                "tgt_text": sentence.values[1],
                "tgt_subword": tgt_subword,
                "error": pred
            }, ensure_ascii=False)

        for sentence, tgt_subword, pred in zip(batch.sentences, batch.tgt,
                                               ged_preds):
            tgt_subword = self.TGT.tokenize.convert_ids_to_tokens(tgt_subword, skip_special_tokens=True)
            sentence.repr_fn = partial(json_repr, tgt_subword=tgt_subword,
                                       pred=pred)

        # # some thing to convert ged_preds to json
        # import pdb
        # pdb.set_trace()

        return batch

    def error_label_factorize(self, errors):
        return sum(
            [[(i, i + 1, e) for e in eb.split("::")]
                for i, eb in enumerate(errors) if eb not in {'CORRECT', NUL}],
            [])

    @classmethod
    def build(cls, path, min_freq=2, fix_len=20, **kwargs):
        r"""
        Build a brand-new Parser, including initialization of all data fields and model parameters.

        Args:
            path (str):
                The path of the model to be saved.
            min_freq (str):
                The minimum frequency needed to include a token in the vocabulary. Default: 2.
            fix_len (int):
                The max length of all subword pieces. The excess part of each piece will be truncated.
                Required if using CharLSTM/BERT.
                Default: 20.
            kwargs (dict):
                A dict holding the unconsumed arguments.
        """

        args = Config(**locals())
        os.makedirs(os.path.dirname(path) or './', exist_ok=True)
        if os.path.exists(path) and not args.build:
            return cls.load(**args)

        state = torch.load(args.checkpoint_path, map_location='cpu')
        args = state['args'].update(args)

        if args.bin_path is None:
            bin = os.path.join(os.path.dirname(args.path), 'bin')
        else:
            bin = args.bin_path
        fbin = os.path.join(bin, 'transform') + '.pt'
        if args.cache and os.path.exists(fbin):
            transform = torch.load(fbin, map_location='cpu')
        else:
            transform = state['transform']
            t = transform.SRC.tokenize

            SRC = Field('src',
                        pad=t.pad,
                        unk=t.unk,
                        bos=t.bos,
                        eos=t.eos,
                        tokenize=t)
            TGT = Field('tgt',
                        pad=t.pad,
                        unk=t.unk,
                        bos=t.bos,
                        eos=t.eos,
                        tokenize=t)
            SRC_ERROR_RAW = RawField('src_error_raw')
            SRC_ERROR = Field('src_error')
            TGT_ERROR_RAW = RawField('tgt_error_raw')
            TGT_ERROR = Field('tgt_error')

            transform = Tree(SRC=SRC,
                             TGT=TGT,
                             SRCERROR=(SRC_ERROR_RAW, SRC_ERROR),
                             TGTERROR=(TGT_ERROR_RAW, TGT_ERROR),
                             error_schema=args.error_schema)

            train = Dataset(transform, args.train, **args)
            # share the vocab
            SRC.vocab = TGT.vocab = t.vocab
            SRC_ERROR = SRC_ERROR.build(train)
            TGT_ERROR = TGT_ERROR.build(train)
            SRC_ERROR.vocab = TGT_ERROR.vocab.update(SRC_ERROR.vocab)
            logger.info(f"{transform}")
            if args.cache:
                os.makedirs(bin, exist_ok=True)
                torch.save(transform, fbin, pickle_module=dill)

        SRC = transform.SRC
        (_, TGT_ERROR) = transform.TGTERROR

        args.update({
            'n_words': len(SRC.vocab),
            'n_labels': len(TGT_ERROR.vocab),
            'pad_index': SRC.pad_index,
            'unk_index': SRC.unk_index,
            'bos_index': SRC.bos_index,
            'eos_index': SRC.eos_index,
            'correct_index': TGT_ERROR.vocab['CORRECT'],
        })
        if "partial" in args.error_schema:
            args.update({
                'nul_index': TGT_ERROR.vocab[NUL],
            })
        logger.info("Building the model")
        model = cls.MODEL(**args)
        if args.gec_init:
            logger.info("Init the model with gec params")
            model.load_pretrained(state['pretrained'])
            model.load_state_dict(state['state_dict'], False)
        else:
            logger.info("Original Bart params")
        logger.info(f"{model}\n")

        parser = cls(args, model, transform)
        parser.model.to(parser.device)
        return parser


class Seq2seqIntervenedParser(Parser):

    def __init__(self, args, gec_model, transform):
        self.gec_model = gec_model
        self.args = gec_model.args.update(args)
        self.transform = transform

        if self.args.lm_alpha > 0:
            from transformers import AutoTokenizer, GPT2LMHeadModel
            self.lm_model = GPT2LMHeadModel.from_pretrained(
                self.args.lm_path).to(self.device)
            self.lm_model.eval()
            gpt2_tokenizer = AutoTokenizer.from_pretrained(self.args.lm_path)

            if self.args.lm_path == "IDEA-CCNL/Wenzhong2.0-GPT2-110M-BertTokenizer-chinese":
                token_id_map = map_token_ids(transform.SRC.vocab,
                                             gpt2_tokenizer.get_vocab(),
                                             {'[UNK]': '[UNK]'})
            else:
                token_id_map = map_token_ids(
                    transform.SRC.vocab, gpt2_tokenizer.get_vocab(), {
                        '<s>': '<|endoftext|>',
                        '</s>': '<|endoftext|>',
                        '<unk>': '<|endoftext|>'
                    })

            self.token_id_map = torch.full((self.args.n_words, 1),
                                           self.args.unk_index,
                                           dtype=torch.long,
                                           device=self.device)
            for i, j in token_id_map:
                self.token_id_map[i] = j

        if self.args.ged_alpha > 0:
            ged_state = torch.load(self.args.ged_path, map_location='cpu')
            ged_args = ged_state['args']
            ged_model = Seq2SeqDetectModel(**ged_args)
            ged_model.load_pretrained(ged_state['pretrained'])
            ged_model.load_state_dict(ged_state['state_dict'], False)
            self.ged_model = ged_model.to(self.device)
            self.ged_model.eval()
            self.args = gec_model.args.update(ged_args).update(args)
            self.transform = ged_state['transform']
            self.SRC = self.transform.SRC
            self.TGT = self.transform.TGT
            (_, self.TGT_ERROR) = self.transform.TGTERROR
        else:
            self.ged_model = None
            self.SRC = self.transform.SRC
            self.TGT = self.transform.TGT
            self.TGT_ERROR = None


    def predict(self,
                data: Union[str, Iterable],
                pred: str = None,
                lang: str = None,
                prob: bool = False,
                batch_size: int = 5000,
                buckets: int = 8,
                workers: int = 0,
                cache: bool = False,
                verbose: bool = True,
                **kwargs):
        r"""
        Args:
            data (Union[str, Iterable]):
                The data for prediction.
                - a filename. If ends with `.txt`, the parser will seek to make predictions line by line from plain texts.
                - a list of instances.
            pred (str):
                If specified, the predicted results will be saved to the file. Default: ``None``.
            lang (str):
                Language code (e.g., ``en``) or language name (e.g., ``English``) for the text to tokenize.
                ``None`` if tokenization is not required.
                Default: ``None``.
            prob (bool):
                If ``True``, outputs the probabilities. Default: ``False``.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 8.
            workers (int):
                The number of subprocesses used for data loading. 0 means only the main process. Default: 0.
            amp (bool):
                Specifies whether to use automatic mixed precision. Default: ``False``.
            cache (bool):
                If ``True``, caches the data first, suggested for huge files (e.g., > 1M sentences). Default: ``False``.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.

        Returns:
            A :class:`~supar.utils.Dataset` object containing all predictions if ``cache=False``, otherwise ``None``.
        """

        args = self.args.update(locals())
        init_logger(logger, verbose=args.verbose)

        self.transform.eval()
        if args.prob:
            self.transform.append(Field('probs'))

        logger.info("Loading the data")
        if args.cache:
            args.bin = os.path.join(os.path.dirname(args.path), 'bin')
        if is_dist():
            batch_size = batch_size // dist.get_world_size()
        data = Dataset(self.transform, **args)
        data.build(batch_size, buckets, False, is_dist(), workers)
        logger.info(f"\n{data}")

        logger.info("Making predictions on the data")
        start = datetime.now()
        self.gec_model.eval()
        with tempfile.TemporaryDirectory() as t:
            # we have clustered the sentences by length here to speed up prediction,
            # so the order of the yielded sentences can't be guaranteed
            for batch in progress_bar(data.loader):
                batch = self.pred_step(batch)
                if is_dist() or args.cache:
                    for s in batch.sentences:
                        with open(os.path.join(t, f"{s.index}"), 'w') as f:
                            f.write(str(s) + '\n')
            elapsed = datetime.now() - start

            if is_dist():
                dist.barrier()
            tdirs = gather(t) if is_dist() else (t, )
            if pred is not None and is_master():
                logger.info(f"Saving predicted results to {pred}")
                with open(pred, 'w') as f:
                    # merge all predictions into one single file
                    if is_dist() or args.cache:
                        sentences = (os.path.join(i, s) for i in tdirs
                                     for s in os.listdir(i))
                        for i in progress_bar(
                                sorted(
                                    sentences,
                                    key=lambda x: int(os.path.basename(x)))):
                            with open(i) as s:
                                shutil.copyfileobj(s, f)
                    else:
                        for s in progress_bar(data):
                            f.write(str(s) + '\n')
            # exit util all files have been merged
            if is_dist():
                dist.barrier()
        logger.info(
            f"{elapsed}s elapsed, {len(data) / elapsed.total_seconds():.2f} Sents/s"
        )

        if not cache:
            return data

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (tuple(
                past_state.index_select(0, beam_idx)
                for past_state in layer_past), )
        return reordered_past

    @torch.no_grad()
    def intervened_decode(self,
                           gec_x,
                           ged_x,
                           src_mask):
        batch_size, *_ = gec_x.shape
        beam_size, n_words = self.args.beam_size, self.args.n_words

        # repeat the src inputs beam_size times
        # [batch_size * beam_size, ...]
        gec_x = gec_x.unsqueeze(1).repeat(1, beam_size, 1,
                                          1).view(-1, *gec_x.shape[1:])
        ori_src_mask = src_mask
        src_mask = src_mask.unsqueeze(1).repeat(1, beam_size, 1).view(
            -1, *src_mask.shape[1:])
        # initialize the tgt inputs by <bos>
        # [batch_size * beam_size, seq_len]
        tgt = gec_x.new_full((batch_size * beam_size, 1),
                             self.args.bos_index,
                             dtype=torch.long)
        # [batch_size * beam_size]
        active = src_mask.new_ones(batch_size * beam_size)
        # [batch_size]
        batches = tgt.new_tensor(range(batch_size)) * beam_size
        # accumulated scores
        scores = gec_x.new_full((batch_size, self.args.beam_size),
                                MIN).index_fill_(-1, tgt.new_tensor(0),
                                                 0).view(-1)

        def rank(scores, mask, k, src_mask=None):
            pred_len = mask.sum(-1, keepdim=True)
            scores = scores / pred_len**self.args.length_penalty
            # In Chinese, it may repeatedly generate the same token, to avoid this, we add a penalty
            if src_mask is not None and self.args.language == 'chinese':
                src_len = src_mask.sum(-1, keepdim=True)
                scores[(pred_len > 1.5 * src_len)[:, 0]] = MIN
            return scores.view(batch_size, -1).topk(k, -1)[1]

        # generated past key values for dummy prefix
        past_key_values = self.gec_model.decoder(
            input_ids=torch.full_like(tgt[:, :1], self.args.eos_index),
            attention_mask=torch.ones_like(src_mask[:, :1]),
            encoder_hidden_states=gec_x,
            encoder_attention_mask=src_mask,
            past_key_values=None,
            use_cache=True)[1]

        if self.args.lm_alpha > 0:
            lm_past_key_values = None

        tgt_mask = tgt.ne(self.args.pad_index)
        if self.args.ged_alpha != 0:
            n_labels = self.args.n_labels
            ged_x_2x = ged_x.unsqueeze(1).repeat(1, self.args.ged_topk, 1,
                                                1).view(-1, *ged_x.shape[1:])
            ged_x = ged_x.unsqueeze(1).repeat(1, beam_size, 1,
                                            1).view(-1, *ged_x.shape[1:])
            src_mask_2x = ori_src_mask.unsqueeze(1).repeat(1, self.args.ged_topk,
                                                    1).view(
                                                        -1, *ori_src_mask.shape[1:])
            batches_2x = tgt.new_tensor(range(batch_size)) * self.args.ged_topk

            _, ged_past_key_values = self.ged_model.decoder(
                input_ids=torch.cat(
                    (torch.full_like(tgt[:, :1], self.args.eos_index), tgt),
                    1),
                attention_mask=torch.cat(
                    (torch.ones_like(tgt_mask[:, :1]), tgt_mask), 1),
                encoder_hidden_states=ged_x,
                encoder_attention_mask=src_mask,
                past_key_values=None,
                use_cache=True)[:2]
            errors = torch.tensor([
                self.TGT_ERROR.vocab[n] != 'CORRECT' for n in range(n_labels)
            ]).to(src_mask)

        for t in range(1, min(self.args.max_len + 1,
                              int(1.8 * gec_x.shape[1]))):
            tgt_mask = tgt.ne(self.args.pad_index)
            input_ids = tgt[:, -1:]
            gec_y, new_past_key_values = self.gec_model.decoder(
                input_ids=input_ids,
                attention_mask=torch.cat(
                    (torch.ones_like(tgt_mask[:, :1]), tgt_mask), 1),
                encoder_hidden_states=gec_x,
                encoder_attention_mask=src_mask,
                past_key_values=past_key_values,
                use_cache=True)[:2]
            del past_key_values
            past_key_values = new_past_key_values

            # [n_active, n_words]
            s_y = self.gec_model.classifier(gec_y[:, -1]).log_softmax(-1)
            # only allow finished sequences to get <pad>
            s_y[~active] = MIN

            s_y[~active, self.args.pad_index] = 0
            ori_s_y = s_y.clone()

            # Calculate the Language Model penalty
            if t >= 2 and self.args.lm_alpha > 0:
                gec_entropy = -(s_y * s_y.exp()).sum(
                    -1, keepdim=True) / math.log(s_y.shape[-1])
                # add lm penalty
                # project gec token id to lm token id
                lm_input_ids = embedding(input_ids, self.token_id_map)
                lm_logits, new_lm_past_key_values = self.lm_model(
                    input_ids=lm_input_ids,
                    attention_mask=tgt_mask[:, 1:],
                    past_key_values=lm_past_key_values,
                    use_cache=True)[:2]
                del lm_past_key_values
                lm_past_key_values = new_lm_past_key_values

                s_lm = lm_logits.log_softmax(-1).view(batch_size * beam_size,
                                                      -1)
                lm_entropy = -(s_lm * s_lm.exp()).sum(
                    -1, keepdim=True) / math.log(s_lm.shape[-1])
                s_lm = s_lm.index_select(-1, self.token_id_map.squeeze(-1))
                s_lm[~active] = MIN
                s_lm[~active, self.args.pad_index] = 0

                lm_beta = self.args.lm_beta
                lm_uncertainty = (lm_entropy * lm_beta) + 1
                gec_uncertainty = (gec_entropy * lm_beta) + 1
                lm_coff = self.args.lm_alpha * (
                    gec_uncertainty / lm_uncertainty)

                s_y = s_y - lm_coff * (1 - s_lm.exp())

            # [batch_size * beam_size, n_words]
            scores = scores.unsqueeze(-1) + s_y
            if self.args.ged_alpha == 0:
                topk = beam_size
            else:
                topk = self.args.ged_topk
            # [batch_size, beam_size]
            cands = rank(scores, tgt_mask, topk, src_mask=src_mask)
            # [batch_size * beam_size]
            scores = scores.view(batch_size, -1).gather(-1, cands).view(-1)
            # beams, tokens = cands // n_words, cands % n_words
            beams, tokens = cands.div(
                n_words, rounding_mode='floor'), (cands % n_words).view(-1, 1)
            indices = (batches.unsqueeze(-1) + beams).view(-1)
            # [batch_size * beam_size, seq_len + 1]
            tgt = torch.cat((tgt[indices], tokens), 1)
            past_key_values = self._reorder_cache(past_key_values, indices)
            if self.args.lm_alpha > 0 and lm_past_key_values is not None:
                lm_past_key_values = self._reorder_cache(
                    lm_past_key_values, indices)
            if self.args.ged_alpha != 0:
                ged_past_key_values = self._reorder_cache(
                    ged_past_key_values, indices)
            active = tokens.ne(
                tokens.new_tensor(
                    (self.args.eos_index, self.args.pad_index))).all(-1)

            if not active.any():
                break

            if self.args.ged_alpha == 0:
                continue

            # following is the ged penalty part
            s_y = ori_s_y

            tgt_mask = tgt.ne(self.args.pad_index)
            ged_y_t, new_ged_past_key_values = self.ged_model.decoder(
                input_ids=tokens,
                attention_mask=torch.cat(
                    (torch.ones_like(tgt_mask[:, :1]), tgt_mask), 1),
                encoder_hidden_states=ged_x_2x,
                encoder_attention_mask=src_mask_2x,
                past_key_values=ged_past_key_values,
                use_cache=True)[:2]
            del ged_past_key_values
            ged_past_key_values = new_ged_past_key_values
            ged_y = ged_y_t[active]

            # calculate the ged coefficient
            error_logits = self.ged_model.error_classifier(ged_y[:, -1])
            s_error = error_logits.log_softmax(-1)
            ged_entropy = -(s_error * s_error.exp()).sum(-1) / math.log(
                s_error.shape[-1])

            s_y = s_y[indices][active]
            gec_entropy = -(s_y * s_y.exp()).sum(-1) / math.log(
                s_y.shape[-1])
            
            ged_beta = self.args.ged_beta
            ged_uncertainty = (ged_entropy * ged_beta) + 1
            gec_uncertainty = (gec_entropy * ged_beta) + 1

            error_coff = gec_uncertainty.new_zeros((batch_size * topk, ))
            error_coff[active] = self.args.ged_alpha * (
                gec_uncertainty / ged_uncertainty)

            # [batch_size * beam_size]
            p_error = error_logits.softmax(-1)
            p_error = p_error[:, errors].sum(-1)

            ged_penalty = error_coff.new_zeros((batch_size * topk, ))
            ged_penalty[active] = p_error

            scores = scores - error_coff * ged_penalty

            cands = scores.view(batch_size, -1).topk(beam_size, -1)[1]
            # [batch_size * beam_size]
            scores = scores.view(batch_size, -1).gather(-1, cands).view(-1)
            indices = (batches_2x.unsqueeze(-1) + cands).view(-1)
            tgt = tgt[indices]
            # reorder the all the past key values
            past_key_values = self._reorder_cache(past_key_values, indices)
            if self.args.lm_alpha > 0 and lm_past_key_values is not None:
                lm_past_key_values = self._reorder_cache(
                    lm_past_key_values, indices)
            ged_past_key_values = self._reorder_cache(ged_past_key_values,
                                                      indices)
            active = tgt[:, -1:].ne(
                tokens.new_tensor(
                    (self.args.eos_index, self.args.pad_index))).all(-1)

        cands = rank(scores.view(-1, 1),
                     tgt.ne(self.args.pad_index),
                     self.args.topk,
                     src_mask=src_mask)

        return tgt[(batches.unsqueeze(-1) + cands).view(-1)].view(
            batch_size, self.args.topk, -1)

    @torch.no_grad()
    def pred_step(self, batch: Batch) -> Batch:
        try:
            src, _ = batch
        except:
            src, = batch
        src_mask = src.ne(self.args.pad_index)

        gec_x = self.gec_model(src)
        if self.ged_model is None:
            ged_x = None
        else:
            ged_x = self.ged_model(src)
        tgt = self.intervened_decode(gec_x, ged_x, src_mask)

        batch.tgt = [[self.TGT.tokenize.decode(cand) for cand in i]
                     for i in tgt.tolist()]
        return batch

    @classmethod
    def load(cls,
             path: str,
             reload: bool = False,
             src: str = 'github',
             checkpoint: bool = False,
             **kwargs) -> Parser:
        r"""
        Loads a parser with data fields and pretrained model parameters.

        Args:
            path (str):
                - a string with the shortcut name of a pretrained model defined in ``supar.MODEL``
                  to load from cache or download, e.g., ``'biaffine-dep-en'``.
                - a local path to a pretrained model, e.g., ``./<path>/model``.
            reload (bool):
                Whether to discard the existing cache and force a fresh download. Default: ``False``.
            src (str):
                Specifies where to download the model.
                ``'github'``: github release page.
                ``'hlt'``: hlt homepage, only accessible from 9:00 to 18:00 (UTC+8).
                Default: ``'github'``.
            checkpoint (bool):
                If ``True``, loads all checkpoint states to restore the training process. Default: ``False``.

        Examples:
            >>> from supar import Parser
            >>> parser = Parser.load('biaffine-dep-en')
            >>> parser = Parser.load('./ptb.biaffine.dep.lstm.char')
        """

        args = Config(**locals())

        gec_state = torch.load(args.gec_path, map_location='cpu')
        gec_args = gec_state['args']
        transform = gec_state['transform']

        gec_model = Seq2SeqModel(**gec_args)
        gec_model.load_pretrained(gec_state['pretrained'])
        gec_model.load_state_dict(gec_state['state_dict'], False)

        parser = cls(args, gec_model, transform)
        parser.gec_model.to(parser.device)
        return parser

