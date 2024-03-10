import IPython.display as ipd
import numpy as np
from datasets import load_dataset


def concatenate(files, chunk_duration=50):

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

    audio_chunk = np.zeros(chunk_duration * sr)

    files = [
        {key: values[i] for key, values in files.items()}
        for i in range(len(files["audio"]))
    ]

    chunk_start_timestamp = files[0]["begin_time"]
    chunk_start = int(chunk_start_timestamp * sr)
    chunk_end = int(chunk_start + chunk_duration * sr)

    speakers = []

    chunk_timestamps_start = []
    chunks_timestamps_end = []

    for element in files:

        timestamp_start = element["begin_time"]
        timestamp_end = element["end_time"]

        samples_start = int(timestamp_start * sr)
        # samples_end = int(timestamp_end * sr)

        audio_segment = element["audio"]["array"]
        audio_length = len(audio_segment)

        speaker = element["speaker_id"]

        start_index = samples_start - chunk_start
        segment_length = min(chunk_end - samples_start, audio_length)

        if samples_start > chunk_end:
            break

        audio_chunk[start_index : start_index + segment_length] = audio_segment[
            :segment_length
        ]

        speakers.append(str(speaker))

        chunk_timestamps_start.append(timestamp_start - chunk_start_timestamp)
        chunks_timestamps_end.append(
            min(timestamp_end - chunk_start_timestamp, chunk_duration)
        )

    new_batch["speakers"].append(speakers)
    new_batch["audio"].append(audio_chunk)
    new_batch["timestamps_start"].append(chunk_timestamps_start)
    new_batch["timestamps_end"].append(chunks_timestamps_end)

    return new_batch


if __name__ == "__main__":

    ds = load_dataset("edinburghcstr/ami", "ihm", split="train")

    dataset = ds.filter(lambda x: x["meeting_id"] == "EN2001a")
    dataset = dataset.sort("begin_time")
    dataset = dataset.select(range(320))
    result = dataset.map(
        concatenate, batched=True, batch_size=32, remove_columns=dataset.column_names
    )

    for i, file in enumerate(result):
        audio = ipd.Audio(file["audio"], rate=16000)
        with open("examples/test_file{}.wav".format(i), "wb") as f:
            f.write(audio.data)
