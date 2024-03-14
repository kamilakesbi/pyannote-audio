from typing import Optional

import torch
from segmentation_model.pyannet_torch import PyanNet
from transformers import PretrainedConfig, PreTrainedModel

from pyannote.audio.utils.loss import binary_cross_entropy
from pyannote.audio.utils.permutation import permutate


class PyanNetConfig(PretrainedConfig):
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

        # if self.specifications.powerset:
        #     # `clamp_min` is needed to set non-speech weight to 1.
        #     class_weight = (
        #         torch.clamp_min(self.model.powerset.cardinality, 1.0)
        #         if self.weigh_by_cardinality
        #         else None
        #     )
        #     seg_loss = nll_loss(
        #         permutated_prediction,
        #         torch.argmax(target, dim=-1),
        #         class_weight=class_weight,
        #         weight=weight,
        #     )
        # else:
        seg_loss = binary_cross_entropy(
            permutated_prediction, target.float(), weight=weight
        )

        return seg_loss
