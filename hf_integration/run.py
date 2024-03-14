import torch
from datasets import load_dataset
from segmentation_model.pretrained_model import PyanNetConfig, SegmentationModel
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments


class SegmentationDataset(Dataset):
    def __init__(self, hf_dataset):
        self.data = hf_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        sample = {
            "input_features": self.data["waveform"][idx],
            "label_ids": self.data["target"][idx],
        }
        return sample


class DataCollator:
    def __call__(self, features):
        batch = {}
        batch["labels"] = torch.tensor([f["label_ids"] for f in features])
        batch["input_features"] = torch.tensor([f["input_features"] for f in features])
        return batch


if __name__ == "__main__":

    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    dataset = load_dataset("kamilakesbi/ami_spd_small_processed")
    print("ok")
    dataset = dataset.remove_columns("label")

    train_dataset = SegmentationDataset(dataset["train"])
    print("ok2")
    training_args = TrainingArguments(
        output_dir="output/",
        per_device_train_batch_size=4,
    )
    print("ok3")
    config = PyanNetConfig()
    model = SegmentationModel(config).to("cuda")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollator(),
    )
    print("ok4")
    trainer.train()
