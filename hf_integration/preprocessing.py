import math

import numpy as np
from datasets import load_dataset

from hf_integration.dataset import concatenate
from pyannote.audio.models.segmentation import PyanNet


def prepare_chunk(file, start_time, duration, max_speakers_per_chunk=3):

    end_time = start_time + duration
    start_frame = math.floor(start_time * sample_rate)
    num_frames = math.floor(duration * sample_rate)
    end_frame = start_frame + num_frames

    waveform = file["audio"][start_frame:end_frame]

    file_annotations = []

    file_labels = []
    for i in range(len(file["timestamps_start"])):

        if file["speakers"][i] not in file_labels:
            file_labels.append(file["speakers"][i])

    for i in range(len(file["timestamps_start"])):

        start_segment = file["timestamps_start"][i]
        end_segment = file["timestamps_end"][i]
        label = file_labels.index(file["speakers"][i])
        file_annotations.append((start_segment, end_segment, label))

    dtype = [("start", "<f4"), ("end", "<f4"), ("labels", "i1")]

    annotations = np.array(file_annotations, dtype)

    chunk_annotations = annotations[
        (annotations["start"] < end_time) & (annotations["end"] > start_time)
    ]

    model = PyanNet(sincnet={"stride": 10})
    step = model.receptive_field.step
    half = 0.5 * model.receptive_field.duration

    start = np.maximum(chunk_annotations["start"], start_time) - start_time - half
    start_idx = np.maximum(0, np.round(start / step)).astype(int)

    end = np.minimum(chunk_annotations["end"], end_time) - start_time - half
    end_idx = np.round(end / step).astype(int)

    labels = list(np.unique(chunk_annotations["labels"]))
    num_labels = len(labels)

    if num_labels > max_speakers_per_chunk:
        pass

    num_frames = model.num_frames(round(duration * model.hparams.sample_rate))
    y = np.zeros((num_frames, num_labels), dtype=np.uint8)

    mapping = {label: idx for idx, label in enumerate(labels)}

    for start, end, label in zip(start_idx, end_idx, chunk_annotations["labels"]):

        mapped_label = mapping[label]
        y[start : end + 1, mapped_label] = 1

    return waveform, y


if __name__ == "__main__":

    ds = load_dataset("edinburghcstr/ami", "ihm", split="train")

    dataset = ds.filter(lambda x: x["meeting_id"] == "EN2001a")
    dataset = dataset.sort("begin_time")
    dataset = dataset.select(range(320))
    result = dataset.map(
        concatenate, batched=True, batch_size=32, remove_columns=dataset.column_names
    )

    sample_rate = 16000
    overlap = 0.25
    start_time = 30
    duration = 10
    file = result[0]

    waveform, y = prepare_chunk(file, start_time, duration, max_speakers_per_chunk=3)
    print(waveform, y)
