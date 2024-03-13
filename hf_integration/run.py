from datasets import load_dataset
from segmentation_model.pretrained_model import PyanNetConfig, SegmentationModel
from torch.utils.data import Dataset
from transformers import DefaultDataCollator, Trainer, TrainingArguments


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


# class DataCollator:
#     def __call__(self, features):

#         batch = {
#         }

#         batch['labels'] = torch.stack([f["label_ids"] for f in features])
#         batch['input_features'] = torch.stack([f["input_features"] for f in features])

#         return batch

if __name__ == "__main__":

    dataset = load_dataset("kamilakesbi/ami_spd_small_processed")

    dataset = dataset.remove_columns("label")

    train_dataset = SegmentationDataset(dataset["train"])

    training_args = TrainingArguments(
        output_dir="./output",
        per_device_train_batch_size=1,
    )

    config = PyanNetConfig()
    model = SegmentationModel(config)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DefaultDataCollator(),
    )
    trainer.train()
