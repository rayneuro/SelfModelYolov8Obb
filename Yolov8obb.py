from buildmodel import *
from trainer import *
from pathlib import Path
from typing import Union ,List




class YoloModel(nn.Module):
    

    def __init__(
        self,
        model: Union[str, Path] = "yolov8n.pt",
        task: str = None,
        verbose: bool = False,
    ) -> None:
        super(YoloModel,self).__init__()

        
        self.model = None
        self.trainer = None
        self.ckpt = None
        

    
    def _check_is_pytorch_model(self) -> None:
        """Raises TypeError is model is not a PyTorch model."""
        pt_str = isinstance(self.model, (str, Path)) and Path(self.model).suffix == ".pt"
        pt_module = isinstance(self.model, nn.Module)
        # if is nn.Moudule 
        if not (pt_module or pt_str):
            raise TypeError(
                f"model='{self.model}' should be a *.pt PyTorch model to run this method, but is a different format. "
                f"PyTorch models can train, val, predict and export, i.e. 'model.train(data=...)', but exported "
                f"formats like ONNX, TensorRT etc. only support 'predict' and 'val' modes, "
                f"i.e. 'yolo predict model=yolov8n.onnx'.\nTo run CUDA or MPS inference please pass the device "
                f"argument directly in your inference command, i.e. 'model.predict(source=..., device=0)'"
            )
        
    
    def train(
        self,
        trainer=None,
        **kwargs,
    ):
        """
        Trains the model using the specified dataset and training configuration.

        This method facilitates model training with a range of customizable settings and configurations. It supports
        training with a custom trainer or the default training approach defined in the method. The method handles
        different scenarios, such as resuming training from a checkpoint, integrating with Ultralytics HUB, and
        updating model and configuration after training.

        When using Ultralytics HUB, if the session already has a loaded model, the method prioritizes HUB training
        arguments and issues a warning if local arguments are provided. It checks for pip updates and combines default
        configurations, method-specific defaults, and user-provided arguments to configure the training process. After
        training, it updates the model and its configurations, and optionally attaches metrics.

        Args:
            trainer (BaseTrainer, optional): An instance of a custom trainer class for training the model. If None, the
                method uses a default trainer. Defaults to None.
            **kwargs (any): Arbitrary keyword arguments representing the training configuration. These arguments are
                used to customize various aspects of the training process.

        Returns:
            (dict | None): Training metrics if available and training is successful; otherwise, None.

        Raises:
            AssertionError: If the model is not a PyTorch model.
            PermissionError: If there is a permission issue with the HUB session.
            ModuleNotFoundError: If the HUB SDK is not installed.
        """

        
        self.ckpt = None
        self.metrics = None  # validation/training metrics
        

        self.trainer = DetectionTrainer()
        
        self.trainer.model =self.trainer.get_model(use_model = True)
        print(self.trainer.model)
        self.model = self.trainer.model

        self._check_is_pytorch_model()

        self.trainer.train()
        # Update model and cfg after training
        
        ckpt = self.trainer.best if self.trainer.best.exists() else self.trainer.last
        self.model, _ = attempt_load_one_weight(ckpt)
        self.overrides = self.model.args
        self.metrics = getattr(self.trainer.validator, "metrics", None)  


        return self.metrics
    
    def val(
        self,
        validator=None,
        **kwargs,
    ):
        """
        Validates the model using a specified dataset and validation configuration.

        This method facilitates the model validation process, allowing for a range of customization through various
        settings and configurations. It supports validation with a custom validator or the default validation approach.
        The method combines default configurations, method-specific defaults, and user-provided arguments to configure
        the validation process. After validation, it updates the model's metrics with the results obtained from the
        validator.

        The method supports various arguments that allow customization of the validation process. For a comprehensive
        list of all configurable options, users should refer to the 'configuration' section in the documentation.

        Args:
            validator (BaseValidator, optional): An instance of a custom validator class for validating the model. If
                None, the method uses a default validator. Defaults to None.
            **kwargs (any): Arbitrary keyword arguments representing the validation configuration. These arguments are
                used to customize various aspects of the validation process.

        Returns:
            (dict): Validation metrics obtained from the validation process.

        Raises:
            AssertionError: If the model is not a PyTorch model.
        """
        custom = {"rect": True}  # method defaults
        args = {**self.overrides, **custom, **kwargs, "mode": "val"}  # highest priority args on the right

        validator = (validator or self._smart_load("validator"))(args=args, _callbacks=self.callbacks)
        validator(model=self.model)
        self.metrics = validator.metrics
        return validator.metrics
    

