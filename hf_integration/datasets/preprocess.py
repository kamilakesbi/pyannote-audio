import math

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset

from pyannote.audio.models.segmentation import PyanNet


def get_labels_in_file(file):
    """Get speakers
    Args:
        file (_type_): _description_

    Returns:
        _type_: _description_
    """

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


def get_start_positions(file, duration, overlap, sample_rate=16000):

    file_duration = len(file["audio"][0]["array"]) / sample_rate
    start_positions = np.arange(0, file_duration, duration * (1 - overlap))

    return start_positions


def get_chunk(file, start_time, duration, max_speakers_per_chunk=3, sample_rate=16000):

    end_time = start_time + duration
    start_frame = math.floor(start_time * sample_rate)
    num_frames = math.floor(duration * sample_rate)
    end_frame = start_frame + num_frames

    waveform = file["audio"][0]["array"][start_frame:end_frame]

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


def pad_target(y, label, max_speakers_per_chunk=3):

    num_speakers = len(label)

    if num_speakers > max_speakers_per_chunk:
        indices = np.argsort(-np.sum(y, axis=0), axis=0)
        y = y[:, indices[:max_speakers_per_chunk]]

    elif num_speakers < max_speakers_per_chunk:
        y = np.pad(
            y,
            ((0, 0), (0, max_speakers_per_chunk - num_speakers)),
            mode="constant",
        )

    return y


def chunk_file(file, duration=2, overlap=0.25):

    new_batch = {
        "waveforms": [],
        "labels": [],
    }

    # TODO: randomize chunk selection

    start_positions = get_start_positions(file, duration, overlap)

    for start_time in start_positions:

        X, y, label = get_chunk(file, start_time, duration)

        new_batch["waveforms"].append(X)

        y = pad_target(y, label)

        new_batch["labels"].append(y)

    return new_batch


def processed_spd_dataset(ds):

    subsets = ["train", "validation", "test"]

    processed_spd_dataset = DatasetDict(
        {
            "train": Dataset.from_dict({}),
            "validation": Dataset.from_dict({}),
            "test": Dataset.from_dict({}),
        }
    )
    for subset in subsets:
        processed_spd_dataset[str(subset)] = ds[str(subset)].map(
            chunk_file,
            batched=True,
            batch_size=1,
            remove_columns=ds[str(subset)].column_names,
        )

    return processed_spd_dataset


if __name__ == "__main__":

    ds = load_dataset("kamilakesbi/ami_spd_small_test")

    processed_dataset = processed_spd_dataset(ds)
    processed_dataset.push_to_hub("kamilakesbi/ami_spd_small_processed")
