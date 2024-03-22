import argparse

import numpy as np
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset


def concatenate(files):

    """_summary_

    Returns:
        _type_: _description_
    """

    new_batch = {
        "audio": [],
        "speakers": [],
        "timestamps_start": [],
        "timestamps_end": [],
    }

    sr = files["audio"][0]["sampling_rate"]

    audio_duration = int(max(files["end_time"]) - files["begin_time"][0])
    # print('Begin, End: ', files['begin_time'][0], max(files['end_time']))

    audio_chunk = np.zeros(audio_duration * sr)

    files = [
        {key: values[i] for key, values in files.items()}
        for i in range(len(files["audio"]))
    ]

    chunk_start_timestamp = files[0]["begin_time"]
    chunk_start = int(chunk_start_timestamp * sr)
    chunk_end = int(chunk_start + audio_duration * sr)

    speakers = []

    chunk_timestamps_start = []
    chunks_timestamps_end = []

    for element in files:

        timestamp_start = element["begin_time"]
        timestamp_end = element["end_time"]

        samples_start = int(timestamp_start * sr)

        audio_segment = element["audio"]["array"]
        audio_length = len(audio_segment)

        speaker = element["speaker_id"]

        start_index = samples_start - chunk_start
        segment_length = min(chunk_end - samples_start, audio_length)

        if samples_start > chunk_end:
            break

        audio_chunk[start_index : start_index + segment_length] += audio_segment[
            :segment_length
        ]

        speakers.append(str(speaker))

        chunk_timestamps_start.append(timestamp_start - chunk_start_timestamp)
        chunks_timestamps_end.append(
            min(timestamp_end - chunk_start_timestamp, audio_duration)
        )

    new_batch["speakers"].append(speakers)

    audio_chunk = {
        "array": audio_chunk,
        "sampling_rate": sr,
    }

    new_batch["audio"].append(audio_chunk)
    new_batch["timestamps_start"].append(chunk_timestamps_start)
    new_batch["timestamps_end"].append(chunks_timestamps_end)

    return new_batch


def create_spd_dataset(
    ds,
    batch_size,
    nb_meetings,
):

    """Function to create a speaker Diarization dataset
    from the ami for ASR.

    Returns:
        ds: _description_
        nb_samples_per_meeting:
        batch_size:
        audio_duration:
    """

    subsets = ["train", "validation", "test"]

    speaker_diarization_dataset = DatasetDict(
        {
            "train": Dataset.from_dict({}),
            "validation": Dataset.from_dict({}),
            "test": Dataset.from_dict({}),
        }
    )

    for subset in subsets:

        meetings = (
            ds[str(subset)]
            .to_pandas()["meeting_id"]
            .unique()[: nb_meetings[str(subset)]]
        )

        concatenate_dataset = Dataset.from_dict(
            {"audio": [], "speakers": [], "timestamps_start": [], "timestamps_end": []}
        )

        print(subset)
        for meeting in meetings:
            print(meeting)
            dataset = ds[str(subset)].filter(
                lambda x: x["meeting_id"] == str(meeting), num_proc=24
            )

            dataset = dataset.sort("begin_time")

            result = dataset.map(
                lambda example: concatenate(example),
                batched=True,
                batch_size=batch_size,
                remove_columns=dataset.column_names,
                num_proc=12,
                # keep_in_memory=True,
            )

            concatenate_dataset = concatenate_datasets([concatenate_dataset, result])

        speaker_diarization_dataset[str(subset)] = concatenate_dataset

    return speaker_diarization_dataset


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", help="", default="-1")
    parser.add_argument("--nb_meetings_train", help="", default="-1")
    parser.add_argument("--nb_meetings_val", help="", default="-1")
    parser.add_argument("--nb_meetings_test", help="", default="-1")

    args = parser.parse_args()

    ds = load_dataset("edinburghcstr/ami", "ihm")

    nb_meetings = {
        "train": int(args.nb_meetings_train),
        "validation": int(args.nb_meetings_val),
        "test": int(args.nb_meetings_test),
    }

    spk_dataset = create_spd_dataset(
        ds,
        batch_size=int(args.bs),
        nb_meetings=nb_meetings,
    )
    spk_dataset.push_to_hub("kamilakesbi/ami_spd_nobatch_full")
