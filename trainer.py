
import gc
import math
import os
import subprocess
import time
import warnings
#from torch_utils import *
from checks import *
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from dataset import *


from dataloader import *
from tensorboard import callbacks
from buildmodel import *
import warnings
import numpy as np
import torch
from torch import distributed as dist
from torch import nn, optim
from utils import *
from tasks import *
from plotting import *
from copy import copy
import sys





SimpleNamespace = type(sys.implementation)


class IterableSimpleNamespace(SimpleNamespace):
    """Ultralytics IterableSimpleNamespace is an extension class of SimpleNamespace that adds iterable functionality and
    enables usage with dict() and for loops.
    """

    def __iter__(self):
        """Return an iterator of key-value pairs from the namespace's attributes."""
        return iter(vars(self).items())

    def __str__(self):
        """Return a human-readable string representation of the object."""
        return "\n".join(f"{k}={v}" for k, v in vars(self).items())

    def __getattr__(self, attr):
        """Custom attribute access error message with helpful information."""
        name = self.__class__.__name__
        raise AttributeError(
            f"""
            '{name}' object has no attribute '{attr}'. This may be caused by a modified or out of date 
            'default.yaml' file
            """
        )

    def get(self, key, default=None):
        """Return the value of the specified key if it exists; otherwise, return the default value."""
        return getattr(self, key, default)



# Default configuration
DEFAULT_CFG_DICT = yaml_load(DEFAULT_CFG_PATH)
for k, v in DEFAULT_CFG_DICT.items():
    if isinstance(v, str) and v.lower() == "none":
        DEFAULT_CFG_DICT[k] = None
DEFAULT_CFG_KEYS = DEFAULT_CFG_DICT.keys()
DEFAULT_CFG = IterableSimpleNamespace(**DEFAULT_CFG_DICT)

def get_save_dir():
    """Return save_dir as created from train/val/predict arguments."""

    return Path('./results')


RANK = -1  # rank of current process



def check_suffix(file="yolov8n.pt", suffix=".pt", msg=""):
    """Check file(s) for acceptable suffix."""
    if file and suffix:
        if isinstance(suffix, str):
            suffix = (suffix,)
        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower().strip()  # file suffix
            if len(s):
                assert s in suffix, f"{msg}{f} acceptable suffix is {suffix}, not {s}"



ROOT = Path('./')  # YOLO

def check_file(file, suffix="", download=True, hard=True):
    """Search/download file (if necessary) and return path."""
    # check_suffix(file, suffix)  # optional
    file = str(file).strip()  # convert to string and strip spaces
    
    
    files = glob.glob(str(ROOT / "**" / file), recursive=True)  # find file
    print('here is files')
    #print(files)
    #print('find files' , files)
    if not files and hard:
        raise FileNotFoundError(f"'{file}' does not exist")
    elif len(files) > 1 and hard:
        raise FileNotFoundError(f"Multiple files match '{file}', specify exact path: {files}")
    return files[0] if len(files) else []  # return file



def check_class_names(names):
    """
    Check class names.

    Map imagenet class codes to human-readable names if required. Convert lists to dicts.
    """
    if isinstance(names, list):  # names is a list
        names = dict(enumerate(names))  # convert to dict
    if isinstance(names, dict):
        # Convert 1) string keys to int, i.e. '0' to 0, and non-string values to strings, i.e. True to 'True'
        names = {int(k): str(v) for k, v in names.items()}
        n = len(names)
        if max(names.keys()) >= n:
            raise KeyError(
                f"{n}-class dataset requires class indices 0-{n - 1}, but you have invalid class indices "
                f"{min(names.keys())}-{max(names.keys())} defined in your dataset YAML."
            )
        if isinstance(names[0], str) and names[0].startswith("n0"):  # imagenet class codes, i.e. 'n01440764'
            names_map = yaml_load('.' / "cfg/datasets/ImageNet.yaml")["map"]  # human-readable names
            names = {k: names_map[v] for k, v in names.items()}
    return names





def check_det_dataset(dataset, autodownload=True):
    """
    Download, verify, and/or unzip a dataset if not found locally.

    This function checks the availability of a specified dataset, and if not found, it has the option to download and
    unzip the dataset. It then reads and parses the accompanying YAML data, ensuring key requirements are met and also
    resolves paths related to the dataset.

    Args:
        dataset (str): Path to the dataset or dataset descriptor (like a YAML file).
        autodownload (bool, optional): Whether to automatically download the dataset if not found. Defaults to True.

    Returns:
        (dict): Parsed dataset information and paths.
    """

    file = check_file(dataset)

    # Download (optional)
    extract_dir = ""
    

    # Read YAML
    data = yaml_load(file, append_filename=True)  # dictionary
    #print(data)
    # Checks
    for k in "train", "val":
        if k not in data:
            if k != "val" or "validation" not in data:
                raise SyntaxError(
                    emojis(f"{dataset} '{k}:' key missing ❌.\n'train' and 'val' are required in all data YAMLs.")
                )
            LOGGER.info("WARNING ⚠️ renaming data YAML 'validation' key to 'val' to match YOLO format.")
            data["val"] = data.pop("validation")  # replace 'validation' key with 'val' key
    if "names" not in data and "nc" not in data:
        raise SyntaxError(emojis(f"{dataset} key missing ❌.\n either 'names' or 'nc' are required in all data YAMLs."))
    if "names" in data and "nc" in data and len(data["names"]) != data["nc"]:
        raise SyntaxError(emojis(f"{dataset} 'names' length {len(data['names'])} and 'nc: {data['nc']}' must match."))
    if "names" not in data:
        data["names"] = [f"class_{i}" for i in range(data["nc"])]
    else:
        data["nc"] = len(data["names"])

    data["names"] = check_class_names(data["names"])

    # Resolve paths
    path = Path(extract_dir or data.get("path") or Path(data.get("yaml_file", "")).parent)  # dataset root
    if not path.is_absolute():
        DATASETS_DIR = './'
        path = (DATASETS_DIR / path).resolve()

    # Set paths
    data["path"] = path  # download scripts
    for k in "train", "val", "test", "minival":
        if data.get(k):  # prepend path
            if isinstance(data[k], str):
                x = (path / data[k]).resolve()
                if not x.exists() and data[k].startswith("../"):
                    x = (path / data[k][3:]).resolve()
                data[k] = str(x)
            else:
                data[k] = [str((path / x).resolve()) for x in data[k]]

    # Parse YAML
    val, s = (data.get(x) for x in ("val", "download"))
    if val:
        val = [Path(x).resolve() for x in (val if isinstance(val, list) else [val])]  # val path
        '''
        if not all(x.exists() for x in val):
            name = clean_url(dataset)  # dataset name with URL auth stripped
            m = f"\nDataset '{name}' images not found ⚠️, missing path '{[x for x in val if not x.exists()][0]}'"
            if s and autodownload:
                LOGGER.warning(m)
            else:
                m += f"\nNote dataset download directory is '{DATASETS_DIR}'. You can update this in '{SETTINGS_YAML}'"
                raise FileNotFoundError(m)
            t = time.time()
            r = None  # success
            
            exec(s, {"yaml": data})
            dt = f"({round(time.time() - t, 1)}s)"
            s = f"success ✅ {dt}, saved to {colorstr('bold', DATASETS_DIR)}" if r in {0, None} else f"failure {dt} ❌"
            LOGGER.info(f"Dataset download {s}\n")
        '''
    #print(data)
    return data  # dictionary





default_cfg = {

    'task': 'obb', # (str) YOLO task, i.e. detect, segment, classify, pose
    'mode': 'train', # (str) YOLO mode, i.e. train, val, predict, export, track, benchmark
    # Train settings -------------------------------------------------------------------------------------------------------
    'model': 'Yolov8l',# (str, optional) path to model file, i.e. yolov8n.pt, yolov8n.yaml
    'data' : './datasets/stomata.yaml' ,# (str, optional) path to data file, i.e. coco8.yaml
    'epochs': 100, # (int) number of epochs to train for
    'time' : 123,#, (float, optional) number of hours to train for, overrides epochs if supplied
    'patience' : 100 ,# (int) epochs to wait for no observable improvement for early stopping of training
    'batch': 16 ,# (int) number of images per batch (-1 for AutoBatch)
    'imgsz': 640 ,# (int | list) input images size as int for train and val modes, or list[w,h] for predict and export modes
    'save': True ,# (bool) save train checkpoints and predict results
    'save_period': -1, # (int) Save checkpoint every x epochs (disabled if < 1)
    'cache': False ,# (bool) True/ram, disk or False. Use cache for data loading
    'device': 'cuda device=0',# (int | str | list, optional) device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu
    'workers': 8, # (int) number of worker threads for data loading (per RANK if DDP)
    'project': 'Stomatadetect',# (str, optional) project name
    'name:' :'Yolov8obb' ,# (str, optional) experiment name, results saved to 'project/name' directory
    'exist_ok': False, # (bool) whether to overwrite existing experiment
    'pretrained': True, # (bool | str) whether to use a pretrained model (bool) or a model to load weights from (str)
    'optimizer': 'AdamW', # (str) optimizer to use, choices=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]
    'verbose': True ,# (bool) whether to print verbose output
    'seed': 0 ,# (int) random seed for reproducibility
    'deterministic': True ,# (bool) whether to enable deterministic mode
    'single_cls': False ,# (bool) train multi-class data as single-class
    'rect': False ,# (bool) rectangular training if mode='train' or rectangular validation if mode='val'
    'cos_lr': False ,# (bool) use cosine learning rate scheduler
    'close_mosaic': 10 ,# (int) disable mosaic augmentation for final epochs (0 to disable)
    'resume': False ,# (bool) resume training from last checkpoint
    'amp': True, # (bool) Automatic Mixed Precision (AMP) training, choices=[True, False], True runs AMP check
    'fraction': 1.0, # (float) dataset fraction to train on (default is 1.0, all images in train set)
    'profile': False, # (bool) profile ONNX and TensorRT speeds during training for loggers
    'freeze': None ,# (int | list, optional) freeze first n layers, or freeze list of layer indices during training
    'multi_scale': False, # (bool) Whether to use multiscale during training
    # Segmentation
    'overlap_mask': True ,# (bool) masks should overlap during training (segment train only)
    'mask_ratio': 4 ,# (int) mask downsample ratio (segment train only)
    # Classification
    'dropout': 0.0 ,# (float) use dropout regularization (classify train only)

    # Val/Test settings ----------------------------------------------------------------------------------------------------
    'val': True ,# (bool) validate/test during training
    'split': 'val', # (str) dataset split to use for validation, i.e. 'val', 'test' or 'train'
    'save_json': False ,# (bool) save results to JSON file
    'save_hybrid': False, # (bool) save hybrid version of labels (labels + additional predictions)
    'conf':  0.5,# (float, optional) object confidence threshold for detection (default 0.25 predict, 0.001 val)
    'iou': 0.7 ,# (float) intersection over union (IoU) threshold for NMS
    'max_det': 300 ,# (int) maximum number of detections per image
    'half': False ,# (bool) use half precision (FP16)
    'dnn': False ,# (bool) use OpenCV DNN for ONNX inference
    'plots': True ,# (bool) save plots and images during train/val

    # Predict settings -----------------------------------------------------------------------------------------------------
    'source': './datasets' ,# (str, optional) source directory for images or videos
    'vid_stride': 1 ,# (int) video frame-rate stride
    'stream_buffer': False, # (bool) buffer all streaming frames (True) or return the most recent frame (False)
    'visualize': False ,# (bool) visualize model features
    'augment': False ,# (bool) apply image augmentation to prediction sources
    'agnostic_nms': False ,# (bool) class-agnostic NMS
    'classes': [0,1,2], # (int | list[int], optional) filter results by class, i.e. classes=0, or classes=[0,2,3]
    'retina_masks': False, # (bool) use high-resolution segmentation masks
    'embed': [None],# (list[int], optional) return feature vectors/embeddings from given layers

    # Visualize settings ---------------------------------------------------------------------------------------------------
    'show': False, # (bool) show predicted images and videos if environment allows
    'save_frames': False, # (bool) save predicted individual video frames
    'save_txt': False ,# (bool) save results as .txt file
    'save_conf': False ,# (bool) save results with confidence scores
    'save_crop': False ,# (bool) save cropped images with results
    'show_labels': True ,# (bool) show prediction labels, i.e. 'person'
    'show_conf': True ,# (bool) show prediction confidence, i.e. '0.99'
    'show_boxes': True ,# (bool) show prediction boxes
    'line_width': None,# (int, optional) line width of the bounding boxes. Scaled to image size if None.

    # Export settings ------------------------------------------------------------------------------------------------------
    'format': 'torchscript', # (str) format to export to
    'keras': False ,# (bool) use Kera=s
    'optimize': False, # (bool) TorchScript: optimize for mobile
    'int8': False ,# (bool) CoreML/TF INT8 quantization
    'dynamic': False, # (bool) ONNX/TF/TensorRT: dynamic axes
    'simplify': False, # (bool) ONNX: simplify model
    'opset': [1,False] ,# (int, optional) ONNX: opset version
    'workspace': 4 , # (int) TensorRT: workspace size (GB)
    'nms': False ,# (bool) CoreML: add NMS

    # Hyperparameters ------------------------------------------------------------------------------------------------------
    'lr0': 0.01, # (float) initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
    'lrf': 0.01, # (float) final learning rate (lr0 * lrf)
    'momentum': 0.937, # (float) SGD momentum/Adam beta1
    'weight_decay': 0.0005, # (float) optimizer weight decay 5e-4
    'warmup_epochs': 3.0 ,# (float) warmup epochs (fractions ok)
    'warmup_momentum': 0.8, # (float) warmup initial momentum
    'warmup_bias_lr': 0.1 ,# (float) warmup initial bias lr
    'box': 7.5 ,# (float) box loss gain
    'cls': 0.5 ,# (float) cls loss gain (scale with pixels)
    'dfl': 1.5 ,# (float) dfl loss gain
    'pose': 12.0, # (float) pose loss gain
    'kobj': 1.0 ,# (float) keypoint obj loss gain
    'label_smoothing': 0.0 ,# (float) label smoothing (fraction)
    'nbs': 64 ,# (int) nominal batch size
    'hsv_h': 0.015, # (float) image HSV-Hue augmentation (fraction)
    'hsv_s': 0.7 ,# (float) image HSV-Saturation augmentation (fraction)
    'hsv_v': 0.4 ,# (float) image HSV-Value augmentation (fraction)
    'degrees': 0.0 ,# (float) image rotation (+/- deg)
    'translate': 0.1 ,# (float) image translation (+/- fraction)
    'scale': 0.5, # (float) image scale (+/- gain)
    'shear': 0.0 ,# (float) image shear (+/- deg)
    'perspective': 0.0 ,# (float) image perspective (+/- fraction), range 0-0.001
    'flipud': 0.0 ,# (float) image flip up-down (probability)
    'fliplr': 0.5, # (float) image flip left-right (probability)
    'bgr': 0.0 ,# (float) image channel BGR (probability)
    'mosaic': 1.0 ,# (float) image mosaic (probability)
    'mixup': 0.0 ,# (float) image mixup (probability)
    'copy_paste': 0.0 ,# (float) segment copy-paste (probability)
    'auto_augment': None ,# (str) auto augmentation policy for classification (randaugment, autoaugment, augmix)
    'erasing': 0.4 ,# (float) probability of random erasing during classification training (0-0.9), 0 means no erasing, must be less than 1.0.
    'crop_fraction': 1.0 ,# (float) image crop fraction for classification (0.1-1), 1.0 means no crop, must be greater than 0.

    # Custom config.yaml ---------------------------------------------------------------------------------------------------
    'cfg': None ,# (str, optional) for overriding defaults.yaml



}




class BaseTrainer:
    """
    BaseTrainer.

    A base class for creating trainers.

    Attributes:
        args (Dict) : Hyperparameters,
        validator (BaseValidator): Validator instance.
        model (nn.Module): Model instance.
        callbacks (defaultdict): Dictionary of callbacks.
        save_dir (Path): Directory to save results.
        wdir (Path): Directory to save weights.
        last (Path): Path to the last checkpoint.
        best (Path): Path to the best checkpoint.
        save_period (int): Save checkpoint every x epochs (disabled if < 1).
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs to train for.
        start_epoch (int): Starting epoch for training.
        device (torch.device): Device to use for training.
        amp (bool): Flag to enable AMP (Automatic Mixed Precision).
        scaler (amp.GradScaler): Gradient scaler for AMP.
        data (str): Path to data.
        trainset (torch.utils.data.Dataset): Training dataset.
        testset (torch.utils.data.Dataset): Testing dataset.
        ema (nn.Module): EMA (Exponential Moving Average) of the model.
        resume (bool): Resume training from a checkpoint.
        lf (nn.Module): Loss function.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        best_fitness (float): The best fitness value achieved.
        fitness (float): Current fitness value.
        loss (float): Current loss value.
        tloss (float): Total loss value.
        loss_names (list): List of loss names.
        csv (Path): Path to results CSV file.
    """
    def __init__(self,  cfg = default_cfg  ,overrides=None, _callbacks=None ):
        """
        Initializes the BaseTrainer class.

        Args:
            cfg (str, optional): Path to a configuration file. Defaults to DEFAULT_CFG.
            overrides (dict, optional): Configuration overrides. Defaults to None.
        """
        
        #self.check_resume(overrides)
        self.args = cfg # cfg is Dict
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')#select_device(self.args.device, self.args.batch)
        else:
            self.device = torch.device('cpu')
        
        
        self.validator = None
        self.metrics = None
        self.plots = {}
        
        init_seeds(self.args['seed'] + 1 , deterministic=self.args['deterministic'])

        # Dirs
        self.save_dir = Path('./results')
        self.args['name'] = self.save_dir.name # update name for loggers
        self.wdir = self.save_dir / "weights"  # weights dir
        if RANK in {-1, 0}:
            self.wdir.mkdir(parents=True, exist_ok=True)  # make dir
            self.args['save_dir'] = str(self.save_dir)
            #yaml_save(self.save_dir / "args.yaml", vars(self.args))  # save run args
            dict2yaml_save(self.save_dir / "args.json", self.args)  # save run args
        self.last, self.best = self.wdir / "last.pt", self.wdir / "best.pt"  # checkpoint paths
        self.save_period = self.args['save_period']

        
        self.batch_size = self.args['batch']
        self.epochs = self.args['epochs']
        self.start_epoch = 0
        if RANK == -1:
            print("RANK = -1 , use GPU cuda:0: ")

        # Device
        if self.device.type in {"cpu", "mps"}:
            self.args['workers'] = 0  # faster CPU training as time dominated by inference, not dataloading

        # Model and Dataset
        self.model = ModelYolov8obb()  # add suffix, i.e. yolov8n -> yolov8n.pt
        self.trainset, self.testset = self.get_dataset()
        self.ema = None

        # Optimization utils init
        self.lf = None
        self.scheduler = None

        # Epoch level metrics
        self.best_fitness = None
        self.fitness = None
        self.loss = None
        self.tloss = None
        self.loss_names = ["Loss"]
        self.csv = self.save_dir / "results.csv"
        self.plot_idx = [0, 1, 2]

        # Callbacks Cancel
        
    


    

    def train(self):
        
        world_size = 1  # default to device 0
       
        self._do_train(world_size)

    def _setup_scheduler(self):
        """Initialize training learning rate scheduler."""
        if self.args['cos_lr']:
            self.lf = one_cycle(1, self.args['lrf'], self.epochs)  # cosine 1->hyp['lrf']
        else:
            self.lf = lambda x: max(1 - x / self.epochs, 0) * (1.0 - self.args['lrf']) + self.args['lrf']  # linear
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)

   

    def _setup_train(self, world_size):
        """Builds dataloaders and optimizer on correct rank process."""

        # Model
        #self.run_callbacks("on_pretrain_routine_start")
        ckpt = self.setup_model()
        self.model = self.model.to(self.device)
        self.set_model_attributes()

        # Freeze layers
        freeze_list = (
            self.args['freeze']
            if isinstance(self.args['freeze'], list)
            else range(self.args['freeze'])
            if isinstance(self.args['freeze'], int)
            else []
        )
        always_freeze_names = [".dfl"]  # always freeze these layers
        freeze_layer_names = [f"model.{x}." for x in freeze_list] + always_freeze_names
        for k, v in self.model.named_parameters():
            # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
            if any(x in k for x in freeze_layer_names):
                LOGGER.info(f"Freezing layer '{k}'")
                v.requires_grad = False
            elif not v.requires_grad and v.dtype.is_floating_point:  # only floating point Tensor can require gradients
                LOGGER.info(
                    f"WARNING ⚠️ setting 'requires_grad=True' for frozen layer '{k}'. "
                    "See ultralytics.engine.trainer for customization of frozen layers."
                )
                v.requires_grad = True

        # Check AMP
        
        
        
            
        '''
        if RANK > -1 and world_size > 1:  # DDP
            dist.broadcast(self.amp, src=0)  # broadcast the tensor from rank 0 to all other ranks (returns None)
        '''
        
        
        

        # Check imgsz
        gs = max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32)  # grid size (max stride)
        self.args['imgsz'] = check_imgsz(self.args['imgsz'], stride=gs, floor=gs, max_dim=1)
        self.stride = gs  # for multiscale training

        # Batch size
        

        # Dataloaders
        batch_size = self.batch_size // max(world_size, 1)
        self.train_loader = self.get_dataloader(self.trainset, batch_size=batch_size, rank=RANK, mode="train")
        if RANK in {-1, 0}:
            # Note: When training DOTA dataset, double batch size could get OOM on images with >2000 objects.
            self.test_loader = self.get_dataloader(
                self.testset, batch_size=batch_size if self.args['task'] == "obb" else batch_size * 2, rank=-1, mode="val"
            )
            self.validator = self.get_validator()
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix="val")
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
            self.ema = ModelEMA(self.model)
            if self.args['plots']:
                self.plot_training_labels()

        # Optimizer
        self.accumulate = max(round(self.args['nbs'] / self.batch_size), 1)  # accumulate loss before optimizing

        weight_decay = self.args['weight_decay'] * self.batch_size * self.accumulate / self.args['nbs']  # scale weight_decay

        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args['nbs'])) * self.epochs

        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.args['optimizer'],
            lr=self.args['lr0'],
            momentum=self.args['momentum'],
            decay=weight_decay,
            iterations=iterations,
        )
        # Scheduler
        self._setup_scheduler()
        self.stopper, self.stop = EarlyStopping(patience=self.args['patience']), False
        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move
        #self.run_callbacks("on_pretrain_routine_end")

    def _do_train(self, world_size=1):
        """Train completed, evaluate and plot if specified by arguments."""
        
        self._setup_train(world_size)

        nb = len(self.train_loader)  # number of batches
        nw = max(round(self.args['warmup_epochs'] * nb), 100) if self.args['warmup_epoachs'] > 0 else -1  # warmup iterations
        last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        #self.run_callbacks("on_train_start")
        LOGGER.info(
            f'Image sizes {640} train, {640} val\n'
            f'Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n'
            f"Logging results to {colorstr('bold', self.save_dir)}\n"
            f'Starting training for ' + (f"{self.args['time']} hours..." if self.args['time'] else f"{self.epochs} epochs...")
        )
        if self.args['close_mosaic']:
            base_idx = (self.epochs - self.args['close_mosaic']) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        epoch = self.start_epoch
        self.optimizer.zero_grad()  # zero any resumed gradients to ensure stability on train start
        while True:
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
                self.scheduler.step()

            self.model.train()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            # Update dataloader attributes (optional)
            if epoch == (self.epochs - self.args['close_mosaic']):
                self._close_dataloader_mosaic()
                self.train_loader.reset()

            
            LOGGER.info(self.progress_string())
            pbar = TQDM(enumerate(self.train_loader), total=nb)
            self.tloss = None
            for i, batch in pbar:
                self.run_callbacks("on_train_batch_start")
                # Warmup
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args['nbs'] / self.batch_size]).round()))
                    for j, x in enumerate(self.optimizer.param_groups):
                        # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x["lr"] = np.interp(
                            ni, xi, [self.args['warmup_bias_lr'] if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.args['warmup_momentum'], self.args['momentum']])

                # Forward
                with torch.cuda.amp.autocast(self.amp):
                    batch = self.preprocess_batch(batch)
                    self.loss, self.loss_items = self.model(batch)
                    
                    self.tloss = (
                        (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None else self.loss_items
                    )

                # Backward
                self.scaler.scale(self.loss).backward()

                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                    # Timed stopping
                    if self.args['time']:
                        self.stop = (time.time() - self.train_time_start) > (self.args['time'] * 3600)
                        if RANK != -1:  # if DDP training
                            broadcast_list = [self.stop if RANK == 0 else None]
                            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                            self.stop = broadcast_list[0]
                        if self.stop:  # training time exceeded
                            break

                # Log
                mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
                loss_len = self.tloss.shape[0] if len(self.tloss.shape) else 1
                losses = self.tloss if loss_len > 1 else torch.unsqueeze(self.tloss, 0)
                if RANK in {-1, 0}:
                    pbar.set_description(
                        ("%11s" * 2 + "%11.4g" * (2 + loss_len))
                        % (f"{epoch + 1}/{self.epochs}", mem, *losses, batch["cls"].shape[0], batch["img"].shape[-1])
                    )
                    self.run_callbacks("on_batch_end")
                    if self.args['plots'] and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

                self.run_callbacks("on_train_batch_end")

            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers
            self.run_callbacks("on_train_epoch_end")
            if RANK in {-1, 0}:
                final_epoch = epoch + 1 >= self.epochs
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

                # Validation
                if self.args['val'] or final_epoch or self.stopper.possible_stop or self.stop:
                    self.metrics, self.fitness = self.validate()
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                if self.args['time']:
                    self.stop |= (time.time() - self.train_time_start) > (self.args['time'] * 3600)

                # Save model
                if self.args['save'] or final_epoch:
                    self.save_model()
                    self.run_callbacks("on_model_save")

            # Scheduler
            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            if self.args['time']:
                mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                self.epochs = self.args['epochs'] = math.ceil(self.args['time'] * 3600 / mean_epoch_time)
                self._setup_scheduler()
                self.scheduler.last_epoch = self.epoch  # do not move
                self.stop |= epoch >= self.epochs  # stop if exceeded epochs
            
            gc.collect()
            torch.cuda.empty_cache()  # clear GPU memory at end of epoch, may help reduce CUDA out of memory errors

            # Early Stopping
            if RANK != -1:  # if DDP training
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                self.stop = broadcast_list[0]
            if self.stop:
                break  # must break all DDP ranks
            epoch += 1

        if RANK in {-1, 0}:
            # Do final val with best.pt
            LOGGER.info(
                f"\n{epoch - self.start_epoch + 1} epochs completed in "
                f"{(time.time() - self.train_time_start) / 3600:.3f} hours."
            )
            self.final_eval()
            if self.args['plots']:
                self.plot_metrics()
            
        gc.collect()
        torch.cuda.empty_cache()
        




    def save_model(self):
        """Save model training checkpoints with additional metadata."""
        import io

        import pandas as pd  # scope for faster 'import ultralytics'

        # Serialize ckpt to a byte buffer once (faster than repeated torch.save() calls)
        buffer = io.BytesIO()
        torch.save(
            {
                "epoch": self.epoch,
                "best_fitness": self.best_fitness,
                "model": None,  # resume and final checkpoints derive from EMA
                "ema": deepcopy(self.ema.ema).half(),
                "updates": self.ema.updates,
                "optimizer": convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
                "train_args": vars(self.args),  # save as dict
                "train_metrics": {**self.metrics, **{"fitness": self.fitness}},
                "train_results": {k.strip(): v for k, v in pd.read_csv(self.csv).to_dict(orient="list").items()},
                "date": datetime.now().isoformat(),
                
            },
            buffer,
        )
        serialized_ckpt = buffer.getvalue()  # get the serialized content to save

        # Save checkpoints
        self.last.write_bytes(serialized_ckpt)  # save last.pt
        if self.best_fitness == self.fitness:
            self.best.write_bytes(serialized_ckpt)  # save best.pt
        if (self.save_period > 0) and (self.epoch > 0) and (self.epoch % self.save_period == 0):
            (self.wdir / f"epoch{self.epoch}.pt").write_bytes(serialized_ckpt)  # save epoch, i.e. 'epoch3.pt'

    def get_dataset(self):
        """
        Get train, val path from data dict if it exists.

        Returns None if data format is not recognized.
        """
        try:
            
            data = check_det_dataset(self.args['data'])
            if "yaml_file" in data:
                    self.args['data'] = data["yaml_file"]  # for validating 'yolo train data=url.zip' usage
        except Exception as e:
            raise RuntimeError(emojis(f"Dataset  error ❌ {e}")) from e
        self.data = data
        print('get_dataset train , test ,val')
        print(data)
        return data["train"], data.get("val") or data.get("test")

    

    def setup_model(self):
        """Load/create/download model for any task."""
        if isinstance(self.model, torch.nn.Module) or self.load_address == None:  # if model is loaded beforehand. No setup needed
            return

        torch.load(self.load_address, map_location=self.device)  # load FP32 model

    def optimizer_step(self):
        """Perform a single step of the training optimizer with gradient clipping and EMA update."""
        self.scaler.unscale_(self.optimizer)  # unscale gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # clip gradients
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)

    def preprocess_batch(self, batch):
        """Allows custom preprocessing model inputs and ground truths depending on task type."""
        return batch

    def validate(self):
        """
        Runs validation on test set using self.validator.

        The returned dict is expected to contain "fitness" key.
        """
        metrics = self.validator(self)
        fitness = metrics.pop("fitness", -self.loss.detach().cpu().numpy())  # use loss as fitness measure if not found
        if not self.best_fitness or self.best_fitness < fitness:
            self.best_fitness = fitness
        return metrics, fitness

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Get model and raise NotImplementedError for loading cfg files."""
        raise NotImplementedError("This task trainer doesn't support loading cfg files")

    def get_validator(self):
        """Returns a NotImplementedError when the get_validator function is called."""
        raise NotImplementedError("get_validator function not implemented in trainer")

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Returns dataloader derived from torch.data.Dataloader."""
        raise NotImplementedError("get_dataloader function not implemented in trainer")

    def build_dataset(self, img_path, mode="train", batch=None):
        """Build dataset."""
        raise NotImplementedError("build_dataset function not implemented in trainer")

    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        Returns a loss dict with labelled training loss items tensor.

        Note:
            This is not needed for classification but necessary for segmentation & detection
        """
        return {"loss": loss_items} if loss_items is not None else ["loss"]

    def set_model_attributes(self):
        """To set or update model parameters before training."""
        self.model.names = self.data["names"]

    def build_targets(self, preds, targets):
        """Builds target tensors for training YOLO model."""
        pass

    def progress_string(self):
        """Returns a string describing training progress."""
        return ""

    # TODO: may need to put these following functions into callback
    def plot_training_samples(self, batch, ni):
        """Plots training samples during YOLO training."""
        pass

    def plot_training_labels(self):
        """Plots training labels for YOLO model."""
        pass

    def save_metrics(self, metrics):
        """Saves training metrics to a CSV file."""
        keys, vals = list(metrics.keys()), list(metrics.values())
        n = len(metrics) + 1  # number of cols
        s = "" if self.csv.exists() else (("%23s," * n % tuple(["epoch"] + keys)).rstrip(",") + "\n")  # header
        with open(self.csv, "a") as f:
            f.write(s + ("%23.5g," * n % tuple([self.epoch + 1] + vals)).rstrip(",") + "\n")

    def plot_metrics(self):
        """Plot and display metrics visually."""
        pass

    def on_plot(self, name, data=None):
        """Registers plots (e.g. to be consumed in callbacks)"""
        path = Path(name)
        self.plots[path] = {"data": data, "timestamp": time.time()}

    def final_eval(self):
        """Performs final evaluation and validation for object detection YOLO model."""
        for f in self.last, self.best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is self.best:
                    LOGGER.info(f"\nValidating {f}...")
                    self.validator.args.plots = self.args['plots']
                    self.metrics = self.validator(model=f)
                    self.metrics.pop("fitness", None)
                    print('Model Eval')


    '''
    def check_resume(self, overrides):
        """Check if resume checkpoint exists and update arguments accordingly."""
        resume = self.args.resume
        if resume:
            try:
                exists = isinstance(resume, (str, Path)) and Path(resume).exists()
                last = Path(check_file(resume) if exists else get_latest_run())

                # Check that resume data YAML exists, otherwise strip to force re-download of dataset
                ckpt_args = attempt_load_weights(last).args
                if not Path(ckpt_args["data"]).exists():
                    ckpt_args["data"] = self.args['data']

                resume = True
                self.args = get_cfg(ckpt_args)
                self.args['model'] = self.args.resume = str(last)  # reinstate model
                for k in "imgsz", "batch", "device":  # allow arg updates to reduce memory or update device on resume
                    if k in overrides:
                        setattr(self.args, k, overrides[k])

            except Exception as e:
                raise FileNotFoundError(
                    "Resume checkpoint not found. Please pass a valid checkpoint to resume from, "
                    "i.e. 'yolo train resume model=path/to/last.pt'"
                ) from e
        self.resume = resume'''


    def resume_training(self, ckpt):
        """Resume YOLO training from given epoch and best fitness."""
        if ckpt is None or not self.resume:
            return
        best_fitness = 0.0
        start_epoch = ckpt.get("epoch", -1) + 1
        if ckpt.get("optimizer", None) is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])  # optimizer
            best_fitness = ckpt["best_fitness"]
        if self.ema and ckpt.get("ema"):
            self.ema.ema.load_state_dict(ckpt["ema"].float().state_dict())  # EMA
            self.ema.updates = ckpt["updates"]
        assert start_epoch > 0, (
            f"{self.args['model']} training to {self.epochs} epochs is finished, nothing to resume.\n"
            f"Start a new training without resuming, i.e. 'yolo train model={self.args['model']}'"
        )
        LOGGER.info(f"Resuming training {self.args['model']} from epoch {start_epoch + 1} to {self.epochs} total epochs")
        if self.epochs < start_epoch:
            LOGGER.info(
                f"{self.model} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {self.epochs} more epochs."
            )
            self.epochs += ckpt["epoch"]  # finetune additional epochs
        self.best_fitness = best_fitness
        self.start_epoch = start_epoch
        if start_epoch > (self.epochs - self.args['close_mosaic']):
            self._close_dataloader_mosaic()

    def _close_dataloader_mosaic(self):
        """Update dataloaders to stop using mosaic augmentation."""
        if hasattr(self.train_loader.dataset, "mosaic"):
            self.train_loader.dataset.mosaic = False
        if hasattr(self.train_loader.dataset, "close_mosaic"):
            LOGGER.info("Closing dataloader mosaic")
            self.train_loader.dataset.close_mosaic(hyp=self.args)

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """
        Constructs an optimizer for the given model, based on the specified optimizer name, learning rate, momentum,
        weight decay, and number of iterations.

        Args:
            model (torch.nn.Module): The model for which to build an optimizer.
            name (str, optional): The name of the optimizer to use. If 'auto', the optimizer is selected
                based on the number of iterations. Default: 'auto'.
            lr (float, optional): The learning rate for the optimizer. Default: 0.001.
            momentum (float, optional): The momentum factor for the optimizer. Default: 0.9.
            decay (float, optional): The weight decay for the optimizer. Default: 1e-5.
            iterations (float, optional): The number of iterations, which determines the optimizer if
                name is 'auto'. Default: 1e5.

        Returns:
            (torch.optim.Optimizer): The constructed optimizer.
        """

        g = [], [], []  # optimizer parameter groups
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
        if name == "auto":
            LOGGER.info(
                f"{colorstr('optimizer:')} 'optimizer=auto' found, "
                f"ignoring 'lr0={self.args['lr0']}' and 'momentum={self.args['momentum']}' and "
                f"determining best 'optimizer', 'lr0' and 'momentum' automatically... "
            )
            nc = getattr(model, "nc", 10)  # number of classes
            lr_fit = round(0.002 * 5 / (4 + nc), 6)  # lr0 fit equation to 6 decimal places
            name, lr, momentum = ("SGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
            self.args['warmup_bias_lr'] = 0.0  # no higher than 0.01 for Adam

        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if "bias" in fullname:  # bias (no decay)
                    g[2].append(param)
                elif isinstance(module, bn):  # weight (no decay)
                    g[1].append(param)
                else:  # weight (with decay)
                    g[0].append(param)

        '''
        if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
            optimizer = getattr(optim, name, optim.Adam)(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif name == "RMSProp":
            optimizer = optim.RMSprop(g[2], lr=lr, momentum=momentum)
        elif name == "SGD":
            optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(
                f"Optimizer '{name}' not found in list of available optimizers "
                f"[Adam, AdamW, NAdam, RAdam, RMSProp, SGD, auto]."
                
            )'''
        optimizer = optim.Adam(g[2], lr=lr, betas= (momentum,0.999), weight_decay=0.0)
        
        optimizer.add_param_group({"params": g[0], "weight_decay": decay})  # add g0 with weight_decay
        optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  # add g1 (BatchNorm2d weights)
        LOGGER.info(
            f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups "
            f'{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias(decay=0.0)'
        )
        return optimizer
    



class DetectionTrainer(BaseTrainer):
    """
    A class extending the BaseTrainer class for training based on a detection model.

    Example:
        ```python

        args = dict(model='yolov8n.pt', data='coco8.yaml', epochs=3)
        trainer = DetectionTrainer(overrides=args)
        trainer.train()
        ```
    """

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        # gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "train", stride=32)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Construct and return dataloader."""
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = False
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.args['workers'] if mode == "train" else self.args['workers'] * 2
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)  # return dataloader

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images by scaling and converting to float."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255
        if self.args['multi_scale']:
            imgs = batch["img"]
            sz = (
                random.randrange(self.args['imgsz'] * 0.5, self.args['imgsz'] * 1.5 + self.stride)
                // self.stride
                * self.stride
            )  # size
            sf = sz / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
                ]  # new shape (stretched to gs-multiple)
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs
        return batch

    def set_model_attributes(self):
        """Nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)."""
        # self.args.box *= 3 / nl  # scale to layers
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # scale to classes and layers
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        self.model.nc = self.data["nc"]  # attach number of classes to model
        self.model.names = self.data["names"]  # attach class names to model
        self.model.args = self.args  # attach hyperparameters to model
        # TODO: self.model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc

    
    def get_model(self, weights=None, verbose=True , use_model = False):
        """Return a YOLO detection model."""

        if use_model:
            return ModelYolov8obb(nc=3,ch =3)
        else :
            return torch.load(weights)

    
    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args= copy(self.args), _callbacks=self.callbacks
        )

    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        Returns a loss dict with labelled training loss items tensor.

        Not needed for classification but necessary for segmentation & detection
        """
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        """Returns a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def plot_training_samples(self, batch, ni):
        """Plots training samples with their annotations."""
        plot_images(
            images=batch["img"],
            batch_idx=batch["batch_idx"],
            cls=batch["cls"].squeeze(-1),
            bboxes=batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_metrics(self):
        """Plots metrics from a CSV file."""
        plot_results(file=self.csv, on_plot=self.on_plot)  # save results.png

    def plot_training_labels(self):
        """Create a labeled training plot of the YOLO model."""
        boxes = np.concatenate([lb["bboxes"] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb["cls"] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)





class Colors:
    """

    This class provides methods to work with the Ultralytics color palette, including converting hex color codes to
    RGB values.

    Attributes:
        palette (list of tuple): List of RGB color values.
        n (int): The number of colors in the palette.
        pose_palette (np.ndarray): A specific color palette array with dtype np.uint8.
    """

    def __init__(self):
        """Initialize colors as hex = matplotlib.colors.TABLEAU_COLORS.values()."""
        hexs = (
            "FF3838",
            "FF9D97",
            "FF701F",
            "FFB21D",
            "CFD231",
            "48F90A",
            "92CC17",
            "3DDB86",
            "1A9334",
            "00D4BB",
            "2C99A8",
            "00C2FF",
            "344593",
            "6473FF",
            "0018EC",
            "8438FF",
            "520085",
            "CB38FF",
            "FF95C8",
            "FF37C7",
        )
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)
        self.pose_palette = np.array(
            [
                [255, 128, 0],
                [255, 153, 51],
                [255, 178, 102],
                [230, 230, 0],
                [255, 153, 255],
                [153, 204, 255],
                [255, 102, 255],
                [255, 51, 255],
                [102, 178, 255],
                [51, 153, 255],
                [255, 153, 153],
                [255, 102, 102],
                [255, 51, 51],
                [153, 255, 153],
                [102, 255, 102],
                [51, 255, 51],
                [0, 255, 0],
                [0, 0, 255],
                [255, 0, 0],
                [255, 255, 255],
            ],
            dtype=np.uint8,
        )

    def __call__(self, i, bgr=False):
        """Converts hex color codes to RGB values."""
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        """Converts hex color codes to RGB values (i.e. default PIL order)."""
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))
    

colors = Colors()  # create instance for 'from utils.plots import colors'

def plot_labels(boxes, cls, names=(), save_dir=Path(""), on_plot=None):
    """Plot training labels including class histograms and box statistics."""
    import pandas  # 
    import seaborn  #

    # Filter matplotlib>=3.7.2 warning and Seaborn use_inf and is_categorical FutureWarnings
    warnings.filterwarnings("ignore", category=UserWarning, message="The figure layout has changed to tight")
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Plot dataset labels
    LOGGER.info(f"Plotting labels to {save_dir / 'labels.jpg'}... ")
    nc = int(cls.max() + 1)  # number of classes
    boxes = boxes[:1000000]  # limit to 1M boxes
    x = pandas.DataFrame(boxes, columns=["x", "y", "width", "height"])

    # Seaborn correlogram
    seaborn.pairplot(x, corner=True, diag_kind="auto", kind="hist", diag_kws=dict(bins=50), plot_kws=dict(pmax=0.9))
    plt.savefig(save_dir / "labels_correlogram.jpg", dpi=200)
    plt.close()

    # Matplotlib labels
    ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)[1].ravel()
    y = ax[0].hist(cls, bins=np.linspace(0, nc, nc + 1) - 0.5, rwidth=0.8)
    for i in range(nc):
        y[2].patches[i].set_color([x / 255 for x in colors(i)])
    ax[0].set_ylabel("instances")
    if 0 < len(names) < 30:
        ax[0].set_xticks(range(len(names)))
        ax[0].set_xticklabels(list(names.values()), rotation=90, fontsize=10)
    else:
        ax[0].set_xlabel("classes")
    seaborn.histplot(x, x="x", y="y", ax=ax[2], bins=50, pmax=0.9)
    seaborn.histplot(x, x="width", y="height", ax=ax[3], bins=50, pmax=0.9)

    # Rectangles
    boxes[:, 0:2] = 0.5  # center
    boxes = ops.xywh2xyxy(boxes) * 1000
    img = Image.fromarray(np.ones((1000, 1000, 3), dtype=np.uint8) * 255)
    for cls, box in zip(cls[:500], boxes[:500]):
        ImageDraw.Draw(img).rectangle(box, width=1, outline=colors(cls))  # plot
    ax[1].imshow(img)
    ax[1].axis("off")

    for a in [0, 1, 2, 3]:
        for s in ["top", "right", "left", "bottom"]:
            ax[a].spines[s].set_visible(False)

    fname = save_dir / "labels.jpg"
    plt.savefig(fname, dpi=200)
    plt.close()
    if on_plot:
        on_plot(fname)


