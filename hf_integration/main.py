from datasets import load_dataset
from segmentation_model.pretrained_model import PyanNetConfig, SegmentationModel
from train import DataCollator
from transformers import Trainer, TrainingArguments

if __name__ == "__main__":

    dataset = load_dataset("kamilakesbi/ami_spd_small_processed")

    train_dataset = dataset["train"].with_format("torch")

    training_args = TrainingArguments(
        output_dir="hf_integration/output/",
        per_device_train_batch_size=1,
        dataloader_num_workers=8,
    )

    config = PyanNetConfig()
    model = SegmentationModel(config).to("cuda")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollator(),
    )
    trainer.train()
