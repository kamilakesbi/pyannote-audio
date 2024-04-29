# MIT License
#
# Copyright (c) 2020- CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from functools import cached_property
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Text, Tuple, Union
from urllib.parse import urlparse

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim
import yaml
from huggingface_hub import HfApi, ModelCard, ModelCardData, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
from lightning_fabric.utilities.cloud_io import _load as pl_load
from pyannote.core import SlidingWindow
from pytorch_lightning.utilities.model_summary import ModelSummary
from torch.utils.data import DataLoader

from pyannote.audio import __version__
from pyannote.audio.core.io import Audio
from pyannote.audio.core.task import (
    Problem,
    Specifications,
    Task,
    UnknownSpecificationsError,
)
from pyannote.audio.utils.multi_task import map_with_specifications
from pyannote.audio.utils.version import check_version

CACHE_DIR = os.getenv(
    "PYANNOTE_CACHE",
    os.path.expanduser("~/.cache/torch/pyannote"),
)
HF_PYTORCH_WEIGHTS_NAME = "pytorch_model.bin"
HF_LIGHTNING_CONFIG_NAME = "config.yaml"


# NOTE: needed to backward compatibility to load models trained before pyannote.audio 3.x
class Introspection:
    pass


@dataclass
class Output:
    num_frames: int
    dimension: int
    frames: SlidingWindow


class Model(pl.LightningModule):
    """Base model

    Parameters
    ----------
    sample_rate : int, optional
        Audio sample rate. Defaults to 16kHz (16000).
    num_channels : int, optional
        Number of channels. Defaults to mono (1).
    task : Task, optional
        Task addressed by the model.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        num_channels: int = 1,
        task: Optional[Task] = None,
    ):
        super().__init__()

        assert (
            num_channels == 1
        ), "Only mono audio is supported for now (num_channels = 1)"

        self.save_hyperparameters("sample_rate", "num_channels")

        self.task = task
        self.audio = Audio(sample_rate=self.hparams.sample_rate, mono="downmix")

    @property
    def task(self) -> Task:
        return self._task

    @task.setter
    def task(self, task: Task):
        # reset (cached) properties when task changes
        del self.specifications
        self._task = task

    def build(self):
        # use this method to add task-dependent layers to the model
        # (e.g. the final classification and activation layers)
        pass

    @property
    def specifications(self) -> Union[Specifications, Tuple[Specifications]]:
        if self.task is None:
            try:
                specifications = self._specifications

            except AttributeError as e:
                raise UnknownSpecificationsError(
                    "Model specifications are not available because it has not been assigned a task yet. "
                    "Use `model.task = ...` to assign a task to the model."
                ) from e

        else:
            specifications = self.task.specifications

        return specifications

    @specifications.setter
    def specifications(
        self, specifications: Union[Specifications, Tuple[Specifications]]
    ):
        if not isinstance(specifications, (Specifications, tuple)):
            raise ValueError(
                "Only regular specifications or tuple of specifications are supported."
            )

        durations = set(s.duration for s in specifications)
        if len(durations) > 1:
            raise ValueError("All tasks must share the same (maximum) duration.")

        min_durations = set(s.min_duration for s in specifications)
        if len(min_durations) > 1:
            raise ValueError("All tasks must share the same minimum duration.")

        self._specifications = specifications

    @specifications.deleter
    def specifications(self):
        if hasattr(self, "_specifications"):
            del self._specifications

    def __example_input_array(self, duration: Optional[float] = None) -> torch.Tensor:
        duration = duration or next(iter(self.specifications)).duration
        return torch.randn(
            (
                1,
                self.hparams.num_channels,
                self.audio.get_num_samples(duration),
            ),
            device=self.device,
        )

    @property
    def example_input_array(self) -> torch.Tensor:
        return self.__example_input_array()

    @cached_property
    def receptive_field(self) -> SlidingWindow:
        """(Internal) frames"""

        receptive_field_size = self.receptive_field_size(num_frames=1)
        receptive_field_step = (
            self.receptive_field_size(num_frames=2) - receptive_field_size
        )
        receptive_field_start = (
            self.receptive_field_center(frame=0) - (receptive_field_size - 1) / 2
        )
        return SlidingWindow(
            start=receptive_field_start / self.hparams.sample_rate,
            duration=receptive_field_size / self.hparams.sample_rate,
            step=receptive_field_step / self.hparams.sample_rate,
        )

    def prepare_data(self):
        self.task.prepare_data()

    def setup(self, stage=None):
        if stage == "fit":
            # let the task know about the trainer (e.g for broadcasting
            # cache path between multi-GPU training processes).
            self.task.trainer = self.trainer

        # setup the task if defined (only on training and validation stages,
        # but not for basic inference)
        if self.task:
            self.task.setup(stage)

        # list of layers before adding task-dependent layers
        before = set((name, id(module)) for name, module in self.named_modules())

        # add task-dependent layers (e.g. final classification layer)
        # and re-use original weights when compatible

        original_state_dict = self.state_dict()
        self.build()

        try:
            missing_keys, unexpected_keys = self.load_state_dict(
                original_state_dict, strict=False
            )

        except RuntimeError as e:
            if "size mismatch" in str(e):
                msg = (
                    "Model has been trained for a different task. For fine tuning or transfer learning, "
                    "it is recommended to train task-dependent layers for a few epochs "
                    f"before training the whole model: {self.task_dependent}."
                )
                warnings.warn(msg)
            else:
                raise e

        # move layers that were added by build() to same device as the rest of the model
        for name, module in self.named_modules():
            if (name, id(module)) not in before:
                module.to(self.device)

        # add (trainable) loss function (e.g. ArcFace has its own set of trainable weights)
        if self.task:
            # let task know about the model
            self.task.model = self
            # setup custom loss function
            self.task.setup_loss_func()
            # setup custom validation metrics
            self.task.setup_validation_metric()

        # list of layers after adding task-dependent layers
        after = set((name, id(module)) for name, module in self.named_modules())

        # list of task-dependent layers
        self.task_dependent = list(name for name, _ in after - before)

    def on_save_checkpoint(self, checkpoint):
        # put everything pyannote.audio-specific under pyannote.audio
        # to avoid any future conflicts with pytorch-lightning updates
        checkpoint["pyannote.audio"] = {
            "versions": {
                "torch": torch.__version__,
                "pyannote.audio": __version__,
            },
            "architecture": {
                "module": self.__class__.__module__,
                "class": self.__class__.__name__,
            },
            "specifications": self.specifications,
        }

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]):
        check_version(
            "pyannote.audio",
            checkpoint["pyannote.audio"]["versions"]["pyannote.audio"],
            __version__,
            what="Model",
        )

        check_version(
            "torch",
            checkpoint["pyannote.audio"]["versions"]["torch"],
            torch.__version__,
            what="Model",
        )

        check_version(
            "pytorch-lightning",
            checkpoint["pytorch-lightning_version"],
            pl.__version__,
            what="Model",
        )

        self.specifications = checkpoint["pyannote.audio"]["specifications"]

        # add task-dependent (e.g. final classifier) layers
        self.setup()

    def forward(
        self, waveforms: torch.Tensor, **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        msg = "Class {self.__class__.__name__} should define a `forward` method."
        raise NotImplementedError(msg)

    # convenience function to automate the choice of the final activation function
    def default_activation(self) -> Union[nn.Module, Tuple[nn.Module]]:
        """Guess default activation function according to task specification

            * sigmoid for binary classification
            * log-softmax for regular multi-class classification
            * sigmoid for multi-label classification

        Returns
        -------
        activation : (tuple of) nn.Module
            Activation.
        """

        def __default_activation(
            specifications: Optional[Specifications] = None,
        ) -> nn.Module:
            if specifications.problem == Problem.BINARY_CLASSIFICATION:
                return nn.Sigmoid()

            elif specifications.problem == Problem.MONO_LABEL_CLASSIFICATION:
                return nn.LogSoftmax(dim=-1)

            elif specifications.problem == Problem.MULTI_LABEL_CLASSIFICATION:
                return nn.Sigmoid()

            else:
                msg = "TODO: implement default activation for other types of problems"
                raise NotImplementedError(msg)

        return map_with_specifications(self.specifications, __default_activation)

    # training data logic is delegated to the task because the
    # model does not really need to know how it is being used.
    def train_dataloader(self) -> DataLoader:
        return self.task.train_dataloader()

    # training step logic is delegated to the task because the
    # model does not really need to know how it is being used.
    def training_step(self, batch, batch_idx):
        return self.task.training_step(batch, batch_idx)

    # validation data logic is delegated to the task because the
    # model does not really need to know how it is being used.
    def val_dataloader(self) -> DataLoader:
        return self.task.val_dataloader()

    # validation logic is delegated to the task because the
    # model does not really need to know how it is being used.
    def validation_step(self, batch, batch_idx):
        return self.task.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def __up_to(self, module_name: Text, requires_grad: bool = False) -> List[Text]:
        """Helper function for freeze_up_to and unfreeze_up_to"""

        tokens = module_name.split(".")
        updated_modules = list()

        for name, module in ModelSummary(self, max_depth=-1).named_modules:
            name_tokens = name.split(".")
            matching_tokens = list(
                token
                for token, other_token in zip(name_tokens, tokens)
                if token == other_token
            )

            # if module is A.a.1 & name is A.a, we do not want to freeze the whole A.a module
            # because it might contain other modules like A.a.2 and A.a.3
            if matching_tokens and len(matching_tokens) == len(tokens) - 1:
                continue

            for parameter in module.parameters(recurse=True):
                parameter.requires_grad = requires_grad
            module.train(mode=requires_grad)

            updated_modules.append(name)

            #  stop once we reached the requested module
            if module_name == name:
                break

        if module_name not in updated_modules:
            raise ValueError(f"Could not find module {module_name}")

        return updated_modules

    def freeze_up_to(self, module_name: Text) -> List[Text]:
        """Freeze model up to specific module

        Parameters
        ----------
        module_name : str
            Name of module (included) up to which the model will be frozen.

        Returns
        -------
        frozen_modules : list of str
            List of names of frozen modules

        Raises
        ------
        ValueError when requested module does not exist

        Note
        ----
        The order of modules is the one reported by self.summary("full").
        If your model does not follow a sequential structure, you might
        want to use freeze_by_name for more control.
        """
        return self.__up_to(module_name, requires_grad=False)

    def unfreeze_up_to(self, module_name: Text) -> List[Text]:
        """Unfreeze model up to specific module

        Parameters
        ----------
        module_name : str
            Name of module (included) up to which the model will be unfrozen.

        Returns
        -------
        unfrozen_modules : list of str
            List of names of frozen modules

        Raises
        ------
        ValueError when requested module does not exist

        Note
        ----
        The order of modules is the one reported by self.summary("full").
        If your model does not follow a sequential structure, you might
        want to use freeze_by_name for more control.
        """
        return self.__up_to(module_name, requires_grad=True)

    def __by_name(
        self,
        modules: Union[List[Text], Text],
        recurse: bool = True,
        requires_grad: bool = False,
    ) -> List[Text]:
        """Helper function for freeze_by_name and unfreeze_by_name"""

        updated_modules = list()

        # Force modules to be a list
        if isinstance(modules, str):
            modules = [modules]

        for name in modules:
            module = getattr(self, name)

            for parameter in module.parameters(recurse=True):
                parameter.requires_grad = requires_grad
            module.train(requires_grad)

            # keep track of updated modules
            updated_modules.append(name)

        missing = list(set(modules) - set(updated_modules))
        if missing:
            raise ValueError(f"Could not find the following modules: {missing}.")

        return updated_modules

    def freeze_by_name(
        self,
        modules: Union[Text, List[Text]],
        recurse: bool = True,
    ) -> List[Text]:
        """Freeze modules

        Parameters
        ----------
        modules : list of str, str
            Name(s) of modules to freeze
        recurse : bool, optional
            If True (default), freezes parameters of these modules and all submodules.
            Otherwise, only freezes parameters that are direct members of these modules.

        Returns
        -------
        frozen_modules: list of str
            Names of frozen modules

        Raises
        ------
        ValueError if at least one of `modules` does not exist.
        """

        return self.__by_name(
            modules,
            recurse=recurse,
            requires_grad=False,
        )

    def unfreeze_by_name(
        self,
        modules: Union[List[Text], Text],
        recurse: bool = True,
    ) -> List[Text]:
        """Unfreeze modules

        Parameters
        ----------
        modules : list of str, str
            Name(s) of modules to unfreeze
        recurse : bool, optional
            If True (default), unfreezes parameters of these modules and all submodules.
            Otherwise, only unfreezes parameters that are direct members of these modules.

        Returns
        -------
        unfrozen_modules: list of str
            Names of unfrozen modules

        Raises
        ------
        ValueError if at least one of `modules` does not exist.
        """

        return self.__by_name(modules, recurse=recurse, requires_grad=True)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: Union[Path, Text],
        map_location=None,
        hparams_file: Union[Path, Text] = None,
        strict: bool = True,
        use_auth_token: Union[Text, None] = None,
        cache_dir: Union[Path, Text] = CACHE_DIR,
        **kwargs,
    ) -> "Model":
        """Load pretrained model

        Parameters
        ----------
        checkpoint : Path or str
            Path to checkpoint, or a remote URL, or a model identifier from
            the huggingface.co model hub.
        map_location: optional
            Same role as in torch.load().
            Defaults to `lambda storage, loc: storage`.
        hparams_file : Path or str, optional
            Path to a .yaml file with hierarchical structure as in this example:
                drop_prob: 0.2
                dataloader:
                    batch_size: 32
            You most likely won’t need this since Lightning will always save the
            hyperparameters to the checkpoint. However, if your checkpoint weights
            do not have the hyperparameters saved, use this method to pass in a .yaml
            file with the hparams you would like to use. These will be converted
            into a dict and passed into your Model for use.
        strict : bool, optional
            Whether to strictly enforce that the keys in checkpoint match
            the keys returned by this module’s state dict. Defaults to True.
        use_auth_token : str, optional
            When loading a private huggingface.co model, set `use_auth_token`
            to True or to a string containing your hugginface.co authentication
            token that can be obtained by running `huggingface-cli login`
        cache_dir: Path or str, optional
            Path to model cache directory. Defaults to content of PYANNOTE_CACHE
            environment variable, or "~/.cache/torch/pyannote" when unset.
        kwargs: optional
            Any extra keyword args needed to init the model.
            Can also be used to override saved hyperparameter values.

        Returns
        -------
        model : Model
            Model

        See also
        --------
        torch.load
        """

        # pytorch-lightning expects str, not Path.

        checkpoint = str(checkpoint)
        if hparams_file is not None:
            hparams_file = str(hparams_file)

        # resolve the checkpoint to
        # something that pl will handle
        if os.path.isfile(checkpoint):
            path_for_pl = checkpoint
        elif urlparse(checkpoint).scheme in ("http", "https"):
            path_for_pl = checkpoint
        else:
            # Finally, let's try to find it on Hugging Face model hub
            # e.g. julien-c/voice-activity-detection is a valid model id
            # and  julien-c/voice-activity-detection@main supports specifying a commit/branch/tag.
            if "@" in checkpoint:
                model_id = checkpoint.split("@")[0]
                revision = checkpoint.split("@")[1]
            else:
                model_id = checkpoint
                revision = None

            try:
                path_for_pl = hf_hub_download(
                    model_id,
                    HF_PYTORCH_WEIGHTS_NAME,
                    repo_type="model",
                    revision=revision,
                    library_name="pyannote",
                    library_version=__version__,
                    cache_dir=cache_dir,
                    # force_download=False,
                    # proxies=None,
                    # etag_timeout=10,
                    # resume_download=False,
                    use_auth_token=use_auth_token,
                    # local_files_only=False,
                    # legacy_cache_layout=False,
                )
            except RepositoryNotFoundError:
                print(
                    f"""
Could not download '{model_id}' model.
It might be because the model is private or gated so make
sure to authenticate. Visit https://hf.co/settings/tokens to
create your access token and retry with:

   >>> Model.from_pretrained('{model_id}',
   ...                       use_auth_token=YOUR_AUTH_TOKEN)

If this still does not work, it might be because the model is gated:
visit https://hf.co/{model_id} to accept the user conditions."""
                )
                return None

            # HACK Huggingface download counters rely on config.yaml
            # HACK Therefore we download config.yaml even though we
            # HACK do not use it. Fails silently in case model does not
            # HACK have a config.yaml file.
            try:
                _ = hf_hub_download(
                    model_id,
                    HF_LIGHTNING_CONFIG_NAME,
                    repo_type="model",
                    revision=revision,
                    library_name="pyannote",
                    library_version=__version__,
                    cache_dir=cache_dir,
                    # force_download=False,
                    # proxies=None,
                    # etag_timeout=10,
                    # resume_download=False,
                    use_auth_token=use_auth_token,
                    # local_files_only=False,
                    # legacy_cache_layout=False,
                )

            except Exception:
                pass

        if map_location is None:

            def default_map_location(storage, loc):
                return storage

            map_location = default_map_location

        # obtain model class from the checkpoint
        loaded_checkpoint = pl_load(path_for_pl, map_location=map_location)
        module_name: str = loaded_checkpoint["pyannote.audio"]["architecture"]["module"]
        module = import_module(module_name)
        class_name: str = loaded_checkpoint["pyannote.audio"]["architecture"]["class"]
        Klass = getattr(module, class_name)

        try:
            model = Klass.load_from_checkpoint(
                path_for_pl,
                map_location=map_location,
                hparams_file=hparams_file,
                strict=strict,
                **kwargs,
            )
        except RuntimeError as e:
            if "loss_func" in str(e):
                msg = (
                    "Model has been trained with a task-dependent loss function. "
                    "Set 'strict' to False to load the model without its loss function "
                    "and prevent this warning from appearing. "
                )
                warnings.warn(msg)
                model = Klass.load_from_checkpoint(
                    path_for_pl,
                    map_location=map_location,
                    hparams_file=hparams_file,
                    strict=False,
                    **kwargs,
                )
                return model

            raise e

        return model

    def push_to_hub(
        self,
        repo_id: str,
        commit_message: Optional[str] = None,
        private: Optional[bool] = None,
        token: Optional[Union[bool, str]] = None,
        create_pr: bool = False,
        revision: str = None,
        commit_description: str = None,
        tags: Optional[List[str]] = None,
        **deprecated_kwargs,
    ) -> None:
        """
        Upload the {object_files} to the 🤗 Model Hub.

        Parameters:
            repo_id (`str`):
                The name of the repository you want to push your {object} to. It should contain your organization name
                when pushing to a given organization.
            use_temp_dir (`bool`, *optional*):
                Whether or not to use a temporary directory to store the files saved before they are pushed to the Hub.
                Will default to `True` if there is no directory named like `repo_id`, `False` otherwise.
            commit_message (`str`, *optional*):
                Message to commit while pushing. Will default to `"Upload {object}"`.
            private (`bool`, *optional*):
                Whether or not the repository created should be private.
            token (`bool` or `str`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `huggingface-cli login` (stored in `~/.huggingface`). Will default to `True` if `repo_url`
                is not specified.
            max_shard_size (`int` or `str`, *optional*, defaults to `"5GB"`):
                Only applicable for models. The maximum size for a checkpoint before being sharded. Checkpoints shard
                will then be each of size lower than this size. If expressed as a string, needs to be digits followed
                by a unit (like `"5MB"`). We default it to `"5GB"` so that users can easily load models on free-tier
                Google Colab instances without any CPU OOM issues.
            create_pr (`bool`, *optional*, defaults to `False`):
                Whether or not to create a PR with the uploaded files or directly commit.
            revision (`str`, *optional*):
                Branch to push the uploaded files to.
            commit_description (`str`, *optional*):
                The description of the commit that will be created
            tags (`List[str]`, *optional*):
                List of tags to push on the Hub.
        """

        ignore_metadata_errors = deprecated_kwargs.pop("ignore_metadata_errors", False)

        api = HfApi()

        _ = api.create_repo(
            repo_id, private=private, token=token, exist_ok=True, repo_type="model"
        )

        model_type = str(type(self)).split("'")[1].split(".")[-1]

        with TemporaryDirectory() as tmpdir:

            tmpdir = Path(tmpdir)

            # Save State Dicts:
            checkpoint = {"state_dict": self.state_dict()}
            self.on_save_checkpoint(checkpoint)
            checkpoint["pytorch-lightning_version"] = pl.__version__

            if model_type == "PyanNet":
                checkpoint["hyper_parameters"] = {
                    "sample_rate": self.hparams.sample_rate,
                    "num_channels": self.hparams.num_channels,
                    "sincnet": self.hparams.sincnet,
                    "lstm": self.hparams.lstm,
                    "linear": self.hparams.linear,
                }

            pyannote_checkpoint = Path(tmpdir) / HF_PYTORCH_WEIGHTS_NAME
            torch.save(checkpoint, pyannote_checkpoint)

            # Prepare Config Files and Tags for a PyanNet model
            if model_type == "PyanNet":
                file = {
                    "model": {},
                    "task": {},
                }
                file["model"] = checkpoint["hyper_parameters"]
                file["model"]["_target_"] = str(type(self)).split("'")[1]
                file["task"]["duration"] = self.specifications.duration
                file["task"]["max_speakers_per_chunk"] = len(
                    self.specifications.classes
                )
                file["task"][
                    "max_speakers_per_frame"
                ] = self.specifications.powerset_max_classes

            # Prepare Config Files and Tags for a WeSpeakerResNet34 model:
            elif model_type == "WeSpeakerResNet34":
                file = {
                    "model": {},
                }

                file["model"] = dict(self.hparams)
                file["model"]["_target_"] = str(type(self)).split("'")[1]

            with open(tmpdir / "config.yaml", "w") as outfile:
                yaml.dump(file, outfile, default_flow_style=False)

            # Update model card:
            model_card = create_and_tag_model_card(
                repo_id,
                model_type,
                token=token,
                ignore_metadata_errors=ignore_metadata_errors,
            )
            model_card.save(os.path.join(tmpdir, "README.md"))

            # Push to hub
            return api.upload_folder(
                repo_id=repo_id,
                folder_path=tmpdir,
                use_auth_token=token,
                repo_type="model",
                commit_message=commit_message,
                create_pr=create_pr,
                revision=revision,
                commit_description=commit_description,
            )


def create_and_tag_model_card(
    repo_id: str,
    model_type: str,
    # tags: Optional[List[str]] = None,
    token: Optional[str] = None,
    ignore_metadata_errors: bool = False,
):
    """
    Creates or loads an existing model card and tags it.

    Args:
        repo_id (`str`):
            The repo_id where to look for the model card.
        tags (`List[str]`, *optional*):
            The list of tags to add in the model card
        token (`str`, *optional*):
            Authentication token, obtained with `huggingface_hub.HfApi.login` method. Will default to the stored token.
        ignore_metadata_errors (`str`):
            If True, errors while parsing the metadata section will be ignored. Some information might be lost during
            the process. Use it at your own risk.
    """

    if model_type == "PyanNet":

        tags = [
            "pyannote",
            "pyannote.audio",
            "pyannote-audio-model",
            "audio",
            "voice",
            "speech",
            "speaker",
            "speaker-diarization",
            "speaker-change-detection",
            "speaker-segmentation",
            "voice-activity-detection",
            "overlapped-speech-detection",
            "resegmentation",
        ]
        licence = "mit"

        extra_gated_prompt = "The collected information will help acquire a better knowledge of \
            pyannote.audio userbase and help its maintainers improve it further. Though \
            this model uses MIT license and will always remain open-source, we will \
            occasionnally email you about premium models and paid services around \
            pyannote."
        # extra_gated_fields:
        #     Company/university: text
        # Website: text

    elif model_type == "WeSpeakerResNet34":

        tags = [
            "pyannote",
            "pyannote.audio",
            "pyannote-audio-model",
            "audio",
            "voice",
            "speech",
            "speaker",
            "speaker-recognition",
            "speaker-verification",
            "speaker-identification",
            "speaker-embedding",
            "PyTorch",
            "wespeaker",
        ]
        licence = "cc-by-4.0"
    try:
        # Check if the model card is present on the remote repo
        model_card = ModelCard.load(
            repo_id, token=token, ignore_metadata_errors=ignore_metadata_errors
        )
    except EntryNotFoundError:
        # Otherwise create a simple model card from template
        model_description = "This is the model card of a pyannote model that has been pushed on the Hub. This model card has been automatically generated."
        card_data = ModelCardData(
            tags=[] if tags is None else tags, library_name="pyannote"
        )
        model_card = ModelCard.from_template(
            card_data, model_description=model_description
        )
    extra_gated_prompt = None

    if tags is not None:
        for model_tag in tags:
            if model_tag not in model_card.data.tags:
                model_card.data.tags.append(model_tag)

    if licence is not None:
        model_card.data.licence = licence

    if extra_gated_prompt is not None:
        model_card.data.extra_gated_prompt = extra_gated_prompt

    return model_card
