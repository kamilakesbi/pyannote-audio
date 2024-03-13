from segmentation_model.pyannet_torch import PyanNet
from transformers import PretrainedConfig, PreTrainedModel


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

    def forward(self, input_features, labels=None):

        logits = self.model(input_features)

        if labels is not None:
            loss = 0
            return {"loss": loss, "logits": logits}

        return {"logits": logits}
