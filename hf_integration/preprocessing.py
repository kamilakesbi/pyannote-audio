import math

import numpy as np
from datasets import load_dataset

from hf_integration.dataset import concatenate
from pyannote.audio.models.segmentation import PyanNet


def get_labels_in_file(file):

    file_labels = []
    for i in range(len(file["speakers"][0])):

        if file["speakers"][0][i] not in file_labels:
            file_labels.append(file["speakers"][0][i])

    return file_labels


def get_segments_in_file(file, labels):

    file_annotations = []

    for i in range(len(file["timestamps_start"][0])):

        start_segment = file["timestamps_start"][0][i]
        end_segment = file["timestamps_end"][0][i]
        label = labels.index(file["speakers"][0][i])
        file_annotations.append((start_segment, end_segment, label))

    dtype = [("start", "<f4"), ("end", "<f4"), ("labels", "i1")]

    annotations = np.array(file_annotations, dtype)

    return annotations


def get_chunk(file, start_time, duration, max_speakers_per_chunk=3, sample_rate=16000):

    end_time = start_time + duration
    start_frame = math.floor(start_time * sample_rate)
    num_frames = math.floor(duration * sample_rate)
    end_frame = start_frame + num_frames

    waveform = file["audio"][0][start_frame:end_frame]

    labels = get_labels_in_file(file)

    file_segments = get_segments_in_file(file, labels)

    chunk_segments = file_segments[
        (file_segments["start"] < end_time) & (file_segments["end"] > start_time)
    ]

    model = PyanNet(sincnet={"stride": 10})
    step = model.receptive_field.step
    half = 0.5 * model.receptive_field.duration

    start = np.maximum(chunk_segments["start"], start_time) - start_time - half
    start_idx = np.maximum(0, np.round(start / step)).astype(int)

    end = np.minimum(chunk_segments["end"], end_time) - start_time - half
    end_idx = np.round(end / step).astype(int)

    labels = list(np.unique(chunk_segments["labels"]))
    num_labels = len(labels)

    if num_labels > max_speakers_per_chunk:
        pass

    num_frames = model.num_frames(round(duration * model.hparams.sample_rate))
    y = np.zeros((num_frames, num_labels), dtype=np.uint8)

    mapping = {label: idx for idx, label in enumerate(labels)}

    for start, end, label in zip(start_idx, end_idx, chunk_segments["labels"]):

        mapped_label = mapping[label]
        y[start : end + 1, mapped_label] = 1

    return waveform, y, labels


def get_start_positions(file, duration, overlap, sample_rate=16000):

    file_duration = len(file["audio"][0]) / sample_rate
    start_positions = np.arange(0, file_duration, duration * (1 - overlap))

    return start_positions


def chunk_file(file, duration=2, overlap=0.5):

    new_batch = {
        "input": [],
        "target": [],
        "label": [],
    }

    start_positions = get_start_positions(file, duration, overlap)

    for start_time in start_positions:

        X, y, label = get_chunk(file, start_time, duration)

        new_batch["input"].append(X)
        new_batch["target"].append(y)
        new_batch["label"].append(label)

    return new_batch


if __name__ == "__main__":

    ds = load_dataset("edinburghcstr/ami", "ihm", split="train")

    dataset = ds.filter(lambda x: x["meeting_id"] == "EN2001a")
    dataset = dataset.sort("begin_time")
    dataset = dataset.select(range(32))
    dataset = dataset.map(
        concatenate, batched=True, batch_size=32, remove_columns=dataset.column_names
    )

    processed_dataset = dataset.map(
        chunk_file, batched=True, batch_size=1, remove_columns=dataset.column_names
    )
