import numpy as np
import torch
from datasets import load_dataset
from metrics import Metrics
from pyannote.core import SlidingWindow, SlidingWindowFeature

from pyannote.audio import Inference, Model


def compute_prediction(file, inference):
    audio = torch.tensor(file["audio"]["array"]).unsqueeze(0).to(torch.float32)
    sample_rate = file["audio"]["sampling_rate"]

    input = {"waveform": audio, "sample_rate": sample_rate}

    prediction = inference(input)
    return prediction


def compute_gt(file, inference):

    audio = torch.tensor(file["audio"]["array"]).unsqueeze(0).to(torch.float32)
    sample_rate = file["audio"]["sampling_rate"]

    # Get the number of frames associated to a chunk:
    _, num_frames, _ = inference.model(
        torch.rand((1, int(inference.duration * sample_rate)))
    ).shape
    # compute frame resolution:
    resolution = inference.duration / num_frames

    audio_duration = len(audio[0]) / sample_rate
    num_frames = int(round(audio_duration / resolution))

    labels = list(set(file["speakers"]))

    gt = np.zeros((num_frames, len(labels)), dtype=np.uint8)

    for i in range(len(file["timestamps_start"])):
        start = file["timestamps_start"][i]
        end = file["timestamps_end"][i]
        speaker = file["speakers"][i]
        start_frame = int(round(start / resolution))
        end_frame = int(round(end / resolution))
        speaker_index = labels.index(speaker)

        gt[start_frame:end_frame, speaker_index] += 1

    return gt


test_dataset = load_dataset("kamilakesbi/real_ami_ihm", split="test")

model = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token=True)

inference = Inference(model, step=2.5)

file = test_dataset[0]


def compute_test_metrics(file, inference):

    metric = Metrics(model.specifications)

    gt = compute_gt(file, inference)
    prediction = compute_prediction(file, inference)

    sample_rate = file["audio"]["sampling_rate"]
    _, num_frames, _ = inference.model(
        torch.rand((1, int(inference.duration * sample_rate)))
    ).shape

    resolution = inference.duration / num_frames
    sliding_window = SlidingWindow(start=0, step=resolution, duration=resolution)
    labels = list(set(file["speakers"]))

    reference = SlidingWindowFeature(
        data=gt, labels=labels, sliding_window=sliding_window
    )

    metrics = {"der": 0, "false_alarm": 0, "missed_detection": 0, "confusion": 0}

    for window, pred in prediction:

        reference_window = reference.crop(window, mode="center")
        common_num_frames = min(num_frames, reference_window.shape[0])

        pred = torch.tensor(pred[:common_num_frames]).unsqueeze(0).permute(0, 2, 1)
        target = (
            torch.tensor(reference_window[:common_num_frames])
            .unsqueeze(0)
            .permute(0, 2, 1)
        )

        metric.metrics["der"](pred, target)
        metric.metrics["false_alarm"](pred, target)
        metric.metrics["missed_detection"](pred, target)
        metric.metrics["confusion"](pred, target)

    metrics["der"] = metric.metrics["der"].compute()
    metrics["false_alarm"] = metric.metrics["false_alarm"].compute()
    metrics["missed_detection"] = metric.metrics["missed_detection"].compute()
    metrics["confusion"] = metric.metrics["confusion"].compute()

    return metrics


for file in test_dataset:
    print(compute_test_metrics(file, inference))
