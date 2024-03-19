import os

from datasets import load_dataset
from metrics import Metrics
from segmentation_model.pretrained_model import (
    SegmentationModel,
    SegmentationModelConfig,
)
from train import DataCollator
from transformers import Trainer, TrainingArguments

from pyannote.audio import Inference, Model

if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    dataset = load_dataset("kamilakesbi/ami_spd_medium_processed")

    train_dataset = dataset["train"].with_format("torch")
    eval_dataset = dataset["validation"].with_format("torch")

    training_args = TrainingArguments(
        output_dir="output/",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-3,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=32,
        dataloader_num_workers=8,
        num_train_epochs=3,
        logging_steps=200,
        load_best_model_at_end=True,
        push_to_hub=False,
        save_safetensors=False,
    )

    pretrained = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token=True)
    config = SegmentationModelConfig()
    model = SegmentationModel(config)
    model.from_pyannote_model(pretrained)

    metric = Metrics(model.specifications)

    def test(model, protocol, subset="test"):
        from pyannote.audio.pipelines.utils import get_devices
        from pyannote.audio.utils.metric import DiscreteDiarizationErrorRate
        from pyannote.audio.utils.signal import binarize

        (device,) = get_devices(needs=1)
        metric = DiscreteDiarizationErrorRate()
        files = list(getattr(protocol, subset)())

        inference = Inference(model, device=device)

        for file in files:
            reference = file["annotation"]
            hypothesis = binarize(inference(file))
            uem = file["annotated"]
            _ = metric(reference, hypothesis, uem=uem)

        return abs(metric)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollator(),
        eval_dataset=eval_dataset,
        compute_metrics=metric.der_metric,
    )
    trainer.train()
