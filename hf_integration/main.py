from datasets import load_dataset
from metrics import der_metric
from segmentation_model.pretrained_model import (
    SegmentationModel,
    SegmentationModelConfig,
)
from train import DataCollator
from transformers import Trainer, TrainingArguments

from pyannote.audio import Model

if __name__ == "__main__":

    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    dataset = load_dataset("kamilakesbi/ami_spd_small_processed")

    train_dataset = dataset["train"].with_format("torch")
    eval_dataset = dataset["validation"].with_format("torch")

    training_args = TrainingArguments(
        output_dir="hf_integration/output/",
        per_device_train_batch_size=4,
        dataloader_num_workers=8,
        do_eval=True,
        do_train=True,
    )

    pretrained = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token=True)
    config = SegmentationModelConfig()
    model = SegmentationModel(config)
    model.copy_weights(pretrained)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollator(),
        eval_dataset=eval_dataset,
        compute_metrics=der_metric,
    )
    trainer.train()
    trainer.evaluate()
