import copy
from typing import Optional

import torch
from segmentation_model.pyannet_torch import PyanNet
from transformers import PretrainedConfig, PreTrainedModel

from pyannote.audio.core.task import Problem, Resolution, Specifications
from pyannote.audio.utils.loss import binary_cross_entropy, nll_loss
from pyannote.audio.utils.permutation import permutate
from pyannote.audio.utils.powerset import Powerset


class SegmentationModelConfig(PretrainedConfig):
    model_type = "pyannet"

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)


class SegmentationModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = PyanNet(sincnet={"stride": 10})

        self.duration = 2
        self.max_speakers_per_frame = None
        self.max_speakers_per_chunk = 3
        self.min_duration = 2
        self.warm_up = (0.0, 0.0)

        self.weigh_by_cardinality = False

        self.specifications = Specifications(
            problem=Problem.MULTI_LABEL_CLASSIFICATION
            if self.max_speakers_per_frame is None
            else Problem.MONO_LABEL_CLASSIFICATION,
            resolution=Resolution.FRAME,
            duration=self.duration,
            min_duration=self.min_duration,
            warm_up=self.warm_up,
            classes=[f"speaker#{i+1}" for i in range(self.max_speakers_per_chunk)],
            powerset_max_classes=self.max_speakers_per_frame,
            permutation_invariant=True,
        )
        self.model.specifications = self.specifications
        self.model.build()
        self.setup_loss_func()

    def forward(self, waveforms, labels=None):

        prediction = self.model(waveforms.unsqueeze(1))
        batch_size, num_frames, _ = prediction.shape

        if labels is not None:

            if self.specifications.powerset:
                weight = torch.ones(batch_size, num_frames, 1, device=waveforms.device)

                multilabel = self.model.powerset.to_multilabel(prediction)
                permutated_target, _ = permutate(multilabel, labels)

                permutated_target_powerset = self.model.powerset.to_powerset(
                    permutated_target.float()
                )
                loss = self.segmentation_loss(
                    prediction, permutated_target_powerset, weight=weight
                )

            else:
                permutated_prediction, _ = permutate(labels, prediction)
                loss = self.segmentation_loss(
                    permutated_prediction, labels, weight=weight
                )

            return {"loss": loss, "logits": prediction}

        return {"logits": prediction}

    def setup_loss_func(self):
        if self.specifications.powerset:
            self.model.powerset = Powerset(
                len(self.specifications.classes),
                self.specifications.powerset_max_classes,
            )

    def segmentation_loss(
        self,
        permutated_prediction: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Permutation-invariant segmentation loss

        Parameters
        ----------
        permutated_prediction : (batch_size, num_frames, num_classes) torch.Tensor
            Permutated speaker activity predictions.
        target : (batch_size, num_frames, num_speakers) torch.Tensor
            Speaker activity.
        weight : (batch_size, num_frames, 1) torch.Tensor, optional
            Frames weight.

        Returns
        -------
        seg_loss : torch.Tensor
            Permutation-invariant segmentation loss
        """

        if self.specifications.powerset:
            # `clamp_min` is needed to set non-speech weight to 1.
            class_weight = (
                torch.clamp_min(self.model.powerset.cardinality, 1.0)
                if self.weigh_by_cardinality
                else None
            )
            seg_loss = nll_loss(
                permutated_prediction,
                torch.argmax(target, dim=-1),
                class_weight=class_weight,
                weight=weight,
            )
        else:
            seg_loss = binary_cross_entropy(
                permutated_prediction, target.float(), weight=weight
            )

        return seg_loss

    def copy_weights(self, pretrained):

        self.model.hparams = copy.deepcopy(pretrained.hparams)

        self.model.sincnet = copy.deepcopy(pretrained.sincnet)
        self.model.sincnet.load_state_dict(pretrained.sincnet.state_dict())

        self.model.lstm = copy.deepcopy(pretrained.lstm)
        self.model.lstm.load_state_dict(pretrained.lstm.state_dict())

        self.model.linear = copy.deepcopy(pretrained.linear)
        self.model.linear.load_state_dict(pretrained.linear.state_dict())

        self.model.specifications = copy.deepcopy(pretrained.specifications)

        self.model.classifier = copy.deepcopy(pretrained.classifier)
        self.model.classifier.load_state_dict(pretrained.classifier.state_dict())

        self.model.activation = copy.deepcopy(pretrained.activation)
        self.model.activation.load_state_dict(pretrained.activation.state_dict())

        self.specifications = self.model.specifications
        self.model.build()
        self.setup_loss_func()
