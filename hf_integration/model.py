from segmentation_model.pretrained_model import (
    SegmentationModel,
    SegmentationModelConfig,
)

from pyannote.audio import Model

pretrained = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token=True)

config = SegmentationModelConfig()
model = SegmentationModel(config)
model.from_pyannote_model(pretrained)
seg_model = model.to_pyannote_model()


print("conversion works :)")
