import argparse
import os

from datasets import load_dataset
from metrics import Metrics, test
from pyannote.database import registry
from segmentation.pretrained_model import SegmentationModel, SegmentationModelConfig
from synthetic_datasets.preprocess import preprocess_spd_dataset
from transformers import Trainer, TrainingArguments

from huggingface.collator import DataCollator
from pyannote.audio import Inference, Model
from pyannote.audio.tasks import SpeakerDiarization

if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    parser = argparse.ArgumentParser()
    # dataset arguments:
    parser.add_argument(
        "--load_processed_dataset",
        help="",
        default="kamilakesbi/ami_spd_augmented_test2_processed",
    )
    parser.add_argument("--dataset_name", help="", default="kamilakesbi/ami_spd_bs_32")
    # Preprocess arguments:
    parser.add_argument("--chunk_duration", help="", default="10")

    # Training Arguments:
    parser.add_argument("--learning_rate", help="", default=1e-3)
    parser.add_argument("--batch_size", help="", default=32)
    parser.add_argument("--epochs", help="", default=3)

    # Model Arguments:
    parser.add_argument("--from_pretrained", help="", default=True)

    # Test arguments:
    parser.add_argument("--do_init_eval", help="", default=True)
    parser.add_argument("--do_test", help="", default=True)

    args = parser.parse_args()

    if str(args.load_processed_dataset):

        dataset = load_dataset(str(args.load_processed_dataset), num_proc=12)
    else:
        dataset = load_dataset(str(args.dataset_name), num_proc=12)
        dataset = preprocess_spd_dataset(
            dataset, chunk_duration=str(args.chunk_duration)
        )

    train_dataset = dataset["train"].with_format("torch")
    eval_dataset = dataset["validation"].with_format("torch")
    test_dataset = dataset["test"].with_format("torch")

    config = SegmentationModelConfig()
    model = SegmentationModel(config)

    if str(args.from_pretrained):
        pretrained = Model.from_pretrained(
            "pyannote/segmentation-3.0", use_auth_token=True
        )
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
        num_train_epochs=1,
        logging_steps=200,
        load_best_model_at_end=True,
        push_to_hub=False,
        save_safetensors=False,
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollator(max_speakers_per_chunk=3),
        eval_dataset=eval_dataset,
        compute_metrics=metric.der_metric,
    )

    if str(args.do_init_eval):
        first_eval = trainer.evaluate()
        print("Initial metric values: ", first_eval)
    trainer.train()

    if str(args.do_test):

        registry.load_database(
            "/home/kamil/projects/AMI-diarization-setup/pyannote/database.yml"
        )
        ami = registry.get_protocol("AMI.SpeakerDiarization.mini")

        seg_task = SpeakerDiarization(
            ami, duration=10.0, max_speakers_per_chunk=3, max_speakers_per_frame=2
        )

        test_file = next(ami.test())
        model = model.to_pyannote_model()

        spk_probability = Inference(model, step=2.5)(test_file)

        der_pretrained = test(model=model, protocol=ami, subset="test")
        print(f"Local DER (fine-tuned) = {der_pretrained}%")
