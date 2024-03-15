import copy
from typing import Optional

import torch
from segmentation_model.pyannet_torch import PyanNet
from transformers import PretrainedConfig, PreTrainedModel

from pyannote.audio.utils.loss import binary_cross_entropy
from pyannote.audio.utils.permutation import permutate


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

    def forward(self, waveforms, labels=None):

        prediction = self.model(waveforms.unsqueeze(1))
        batch_size, num_frames, _ = prediction.shape

        if labels is not None:

            weight = torch.ones(batch_size, num_frames, 1, device=waveforms.device)

            permutated_prediction, _ = permutate(labels, prediction)
            loss = self.segmentation_loss(permutated_prediction, labels, weight=weight)

            return {"loss": loss, "logits": prediction}

        return {"logits": prediction}

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
