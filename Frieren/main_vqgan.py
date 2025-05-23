import importlib
import argparse, os, sys, datetime, glob
import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
import time
from pytorch_lightning.utilities import rank_zero_info
import numpy as np
    # model:
    #   base_learning_rate: float
    #   target: path to lightning module
    #   params:
    #       key: value/home/mithilpn/iitgn/dl/final-project-v2/SpecVQGAN/configs
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


class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.root_gpu)
        torch.cuda.synchronize(trainer.root_gpu)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        torch.cuda.synchronize(trainer.root_gpu)
        max_memory = torch.cuda.max_memory_allocated(trainer.root_gpu) / 2 ** 20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.training_type_plugin.reduce(max_memory)
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass


class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            print('Project config')
            # print(self.config.pretty())
            OmegaConf.save(self.config, os.path.join(self.cfgdir, '{}-project.yaml'.format(self.now)))

            print('Lightning config')
            # print(self.lightning_config.pretty())
            OmegaConf.save(OmegaConf.create({'lightning': self.lightning_config}),
                           os.path.join(self.cfgdir, '{}-lightning.yaml'.format(self.now)))

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, 'child_runs', name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None, predict=None,
                 wrap=False, num_workers=None, shuffle_test_loader=False, use_worker_init_fn=False,
                 shuffle_val_dataloader=False):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = self._val_dataloader
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = self._test_dataloader
            # self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config_input_dict(data_cfg)        # !! input dict

    def setup(self, stage=None):            ## Init The Dataset !!
        self.datasets = dict(
            (k, instantiate_from_config_input_dict(self.dataset_configs[k]))      # !! Input params dict
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    @staticmethod
    def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)


    def _train_dataloader(self):
        return DataLoader(self.datasets['train'], batch_size=self.batch_size, num_workers=self.num_workers, worker_init_fn=self.worker_init_fn, shuffle=True)

    def _val_dataloader(self):
        return DataLoader(self.datasets['validation'], batch_size=self.batch_size, num_workers=self.num_workers, worker_init_fn=self.worker_init_fn, shuffle=True)

    def _test_dataloader(self):
        return DataLoader(self.datasets['validation'], batch_size=self.batch_size, num_workers=self.num_workers, worker_init_fn=self.worker_init_fn, shuffle=True)


class WrappedDataset(Dataset):
    '''Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset'''
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


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
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
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
    parser.add_argument(
        "-p",
        "--project",
        help="name of new or path to existing project"
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=72,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )

    parser.add_argument(
        "--stage",
        type=str,
        nargs="?",
        const=True,
        default=True,
        help="stage",
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=2000,
        help="epoch num",
    )

    parser.add_argument(
        "--wandb_project",
        type=str,
        default="",
        help="wandb project",
    )

    return parser

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit('.', 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not 'target' in config:
        raise KeyError('Expected key `target` to instantiate.')
    print(config['target'])
    return get_obj_from_str(config['target'])(**config.get('params', dict()))

def instantiate_from_config_input_dict(config):
    if not 'target' in config:
        raise KeyError('Expected key `target` to instantiate.')
    print(config['target'])
    return get_obj_from_str(config['target'])(config.get('params', dict()))


def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


# Launch the Training:
if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    parser = get_parser()   # --base config_path.yaml --name exper1 --gpus 0, 1, 2
    parser = Trainer.add_argparse_args(parser)
    opt, unknown = parser.parse_known_args()

    stage = opt.stage

    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    
    if opt.resume:  # resume path
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            # logdir = "/".join(paths[:-2])
            logdir = "/".join(paths[:-2])   # /val paths
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")
        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))    # Find the Config File
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""
        # now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        nowname = now + name + opt.postfix
        # nowname = name + opt.postfix
        logdir = os.path.join(opt.logdir, nowname)      # LogDir
    


    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)


    # Init and Save Configs:  # DDP
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    print(cli)  # ?
    config = OmegaConf.merge(*configs, cli)
    lightning_config = config.pop("lightning", OmegaConf.create())
    # merge trainer cli with config
    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    # default to ddp: !!!
    trainer_config['accelerator'] = 'ddp'
    # Non Default trainer Config:
    for k in nondefault_trainer_args(opt):
        trainer_config[k] = getattr(opt, k)
    # Use GPU:
    gpuinfo = trainer_config["gpus"]
    print(f"Running on GPUs {gpuinfo}")
    cpu = False
    trainer_opt = argparse.Namespace(**trainer_config)
    lightning_config.trainer = trainer_config

    # max_epochs=20

    # Model:
    model = instantiate_from_config(config.model)

    # trainer and callbacks:
    trainer_kwargs = dict()
    
    root_dir = os.getcwd()

    # default logger configs:
    if opt.wandb_project:
        project_name = opt.wandb_project
    else:
        project_name = "audio_diffusion"

    default_logger_cfgs = {
            "wandb": {
                "target": "pytorch_lightning.loggers.WandbLogger",
                "params": {
                    "project": project_name,
                    "name": nowname,
                    "save_dir": os.path.join(root_dir, logdir),
                    "id": nowname,
                }
            },
            "testtube": {
                "target": "pytorch_lightning.loggers.TestTubeLogger",
                "params": {
                    "name": "testtube",
                    "save_dir": logdir,
                }
            },
            
            "tensorboard": {
                "target": "pytorch_lightning.loggers.TensorBoardLogger",
                "params":{
                    "name": nowname,
                    "save_dir": logdir,
                }
            },
        }

    # default_logger_cfg = default_logger_cfgs["wandb"]
    default_logger_cfg = default_logger_cfgs["tensorboard"]

    if "logger" in lightning_config:
        logger_cfg = lightning_config.logger
    else:
        logger_cfg = OmegaConf.create()
    
    logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
    trainer_kwargs['logger'] = instantiate_from_config(logger_cfg)

    print(config)
    print('ckptpath',ckptdir)


    default_modelckpt_cfg = {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": os.path.join(root_dir,ckptdir),
                "filename": "{epoch:06d}",
                "verbose": True,
                "save_last": True,
                'save_top_k': 3,
                'period': config.checkpoint.save_every_n_epochs,
            }
    }

    if hasattr(model, "monitor"):
        print("Yes !!!!!!!!!!!!!!!!!!!!!!!! Monitor")
        print(f"Monitoring {model.monitor} as checkpoint metric.")
        print("config.checkpoint.save_every_n_epochs",config.checkpoint.save_every_n_epochs)
        default_modelckpt_cfg["params"]["monitor"] = model.monitor
        print(model.monitor)

    if "modelcheckpoint" in lightning_config:
        modelckpt_cfg = lightning_config.modelcheckpoint
    else:
        modelckpt_cfg = OmegaConf.create()
    modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)

    default_callbacks_cfg = {
            "setup_callback": {
                "target": "main.SetupCallback",
                "params": {
                    "resume": opt.resume,
                    "now": now,
                    "logdir": logdir,
                    "ckptdir": ckptdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": lightning_config,
                }
            }
        }

    
    default_callbacks_cfg[config.callback.logger_name] = dict(config.callback)
    default_callbacks_cfg['cuda_callback'] = {"target": "main.CUDACallback"}
    
    default_callbacks_cfg.update({'checkpoint_callback': modelckpt_cfg})

    if "callbacks" in lightning_config:
        callbacks_cfg = lightning_config.callbacks
    else:
        callbacks_cfg = OmegaConf.create()
    
    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)

    trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]
    trainer_kwargs['max_epochs'] = opt.epoch
    
    trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
    trainer.logdir = logdir  ## logdir

    # Data:
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    print("### Data ###")
    for k in data.datasets:
            print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

    # configure learning rate:
    bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
    if not cpu:
        ngpu = len(lightning_config.trainer.gpus.strip(",").split(','))
    else:
        ngpu = 1
    if 'accumulate_grad_batches' in lightning_config.trainer:
        accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
    else:
        accumulate_grad_batches = 1
    lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches

    if opt.scale_lr:
        model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
        print(
            "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                    model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))
    else:
        model.learning_rate = base_lr
        print("+++ Not Using LR Scaling ++++")
        print(f"Setting learning rate to {model.learning_rate:.2e}")
    

    # Checkpointing:
    def melk(*args, **kwargs):
        # run all checkponit hooks
        if trainer.global_rank == 0:
            print("Summoning checkpoint")
            ckpt_path = os.path.join(ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pudb;
                pudb.set_trace()
    
    import signal
    signal.signal(signal.SIGUSR1, melk)
    signal.signal(signal.SIGUSR2, divein)

    # Run the Model:
    if opt.train:
        try:
            trainer.fit(model, data)
        except Exception:
            melk()
            raise
    
    if not opt.no_test and not trainer.interrupted:
            trainer.test(model, data)
    print("Finishing Training !!")
