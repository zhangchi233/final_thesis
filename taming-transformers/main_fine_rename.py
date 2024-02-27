import argparse, os, sys, datetime, glob, importlib
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
import torch
import torchvision
from torch.utils.data import random_split, DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities import rank_zero_only
import sys

from taming.data.dtu import DTUDataset



def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=False,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="/root/autodl-tmp/taming-transformers/scripts/logs/vqgan_gumbel_f8/checkpoints/last.ckpt",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="/root/autodl-tmp/taming-transformers/scripts/logs/vqgan_gumbel_f8/configs/model.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list("/root/autodl-tmp/taming-transformers/scripts/logs/vqgan_gumbel_f8/configs/model.yaml"),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=True,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument("-p", "--project", help="name of new or path to existing project")
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )

    return parser


def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, root_dir = "/root/autodl-tmp/mvs_training/dtu",train=True,
                  validation=True, test=None,
                 wrap=False, num_workers=None):
        super().__init__()
        self.batch_size = batch_size
        self.datasets = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size*2
        if train is not None:
            self.datasets["train"] = DTUDataset(root_dir, split="train",img_wh=(512,640))
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.datasets["validation"] =  DTUDataset(root_dir, split="val",img_wh=(512,640))
            self.val_dataloader = self._val_dataloader
        if test is not None:
            self.datasets["test"] =  DTUDataset(root_dir, split="test",img_wh=(512,640))
            self.test_dataloader = self._test_dataloader
        



   

    def _train_dataloader(self):
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True)

    def _val_dataloader(self):
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def _test_dataloader(self):
        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers)


class ImageLogger(Callback):

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.log_img(pl_module, batch, batch_idx, split="val")



if __name__ == "__main__":
    # custom parser to specify config files, train, test and debug mode,
    # postfix, resume.
    # `--key value` arguments are interpreted as arguments to the trainer.
    # `nested.key=value` arguments are interpreted as config parameters.
    # configs are merged from left-to-right followed by command line parameters.

    # model:
    #   base_learning_rate: float
    #   target: path to lightning module
    #   params:
    #       key: value
    # data:
    #   target: main.DataModuleFromConfig
    #   params:
    #      batch_size: int
    #      wrap: bool
    #      train:
    #          target: path to train dataset
    #          params:
    #              key: value
    #      validation:
    #          target: path to validation dataset
    #          params:
    #              key: value
    #      test:
    #          target: path to test dataset
    #          params:
    #              key: value
    # lightning: (optional, has sane defaults and can be specified on cmdline)
    #   trainer:
    #       additional arguments to trainer
    #   logger:
    #       logger to instantiate
    #   modelcheckpoint:
    #       modelcheckpoint to instantiate
    #   callbacks:
    #       callback1:
    #           target: importpath
    #           params:
    #               key: value

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    # (in particular `main.DataModuleFromConfig`)
    sys.path.append(os.getcwd())

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args()
    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            # create new log dir\
            print(paths)
            log_path = "/".join(paths[:-2])
            now = f"/logs/"
            log_path = os.path.join(log_path, now)
            os.makedirs(log_path, exist_ok=True)
            
            logdir =log_path 
            
            print(f"New logdir: {logdir}")
            ckpt = opt.resume
            print(f"Resuming from checkpoint: {ckpt}")
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs+opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[_tmp.index("logs")+1]
    else:
        if opt.name:
            name = "_"+opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_"+cfg_name
        else:
            name = ""
        nowname = now+name+opt.postfix
        logdir = os.path.join("logs", nowname)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)

    try:
        # init and save configs
        cfg = "/root/autodl-tmp/taming-transformers/scripts/logs/vqgan_imagenet_f16_16384/configs/model.yaml"
        ckptdir ="/root/autodl-tmp/taming-transformers/scripts/logs/vqgan_imagenet_f16_16384/checkpoints/last.ckpt"
        configs = [OmegaConf.load(cfg)]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)
        config.model.params.ckpt_path = ckptdir

        # model
        model = instantiate_from_config(config.model)
        model.learning_rate = config.model.base_learning_rate
        model.image_key = "imgs"

        # data
    
        data = DataModuleFromConfig(batch_size=1)

        from pytorch_lightning.loggers import TensorBoardLogger

        logger = TensorBoardLogger("/root/autodl-tmp/taming-transformers/scripts/logs/vqgan_gumbel_f8/logs", name=nowname, default_hp_metric=False)
        
        checkpoins = ModelCheckpoint(dirpath=ckptdir, 
                                     save_last=True, save_top_k=1, 
                                     monitor="val/rec_loss", mode="min",
                                     filename="{epoch:02d}-{val/rec_loss:.2f}"
                                     )
        lr_monitor = LearningRateMonitor(logging_interval='step')

        trainer = Trainer(
            gpus=1,
            logger=logger,
    
            callbacks=[checkpoins, lr_monitor],
            max_epochs=20,
            #resume_from_checkpoint=ckptdir,
            progress_bar_refresh_rate=1,
            val_check_interval=1.0,
            
            

        )
        trainer.fit(model, data)
        
        

        
    except Exception:
       
        raise
    
