import torch
from segmentation_model.pretrained_model import (
    SegmentationModel,
    SegmentationModelConfig,
)

from pyannote.audio import Model

pretrained = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token=True)

config = SegmentationModelConfig()
model = SegmentationModel(config)
model.copy_weights(pretrained)

waveform = torch.rand((1, 32000))
