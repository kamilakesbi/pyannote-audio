from datasets import load_dataset
from metrics import Metrics
from segmentation.pretrained_model import SegmentationModel, SegmentationModelConfig
from transformers import Trainer, TrainingArguments
from utils import DataCollator

from pyannote.audio import Model

dataset = load_dataset(str("kamilakesbi/ami_spd_bs_32_processed"), num_proc=12)

train_dataset = dataset["train"].with_format("torch")
eval_dataset = dataset["validation"].with_format("torch")
test_dataset = dataset["test"].with_format("torch")

config = SegmentationModelConfig()
model = SegmentationModel(config)

pretrained = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token=True)
model.from_pyannote_model(pretrained)

metric = Metrics(model.specifications)

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
    seed=42,
)

trainer = Trainer(
    model=model,
    # args=training_args,
    # train_dataset=train_dataset,
    data_collator=DataCollator(max_speakers_per_chunk=3),
    eval_dataset=eval_dataset,
    compute_metrics=metric.der_metric,
)

print("basline: ", trainer.evaluate())
