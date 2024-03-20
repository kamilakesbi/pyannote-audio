import numpy as np
from datasets import load_dataset

common_voice = load_dataset("mozilla-foundation/common_voice_16_1", "en", num_proc=24)


def estimate_audio_duration(batch, sr, audio_file_length=0.9):

    audio_duration = 0
    for row in batch:
        audio_duration += len(row["audio"]["array"]) / sr

    audio_duration *= audio_file_length

    return audio_duration


def concatenate_no_timestamps(files, audio_file_length=1.1, std_concatenate=3):

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

    batch = [
        {key: values[i] for key, values in files.items()}
        for i in range(len(files["audio"]))
    ]

    audio_duration = estimate_audio_duration(batch, sr, audio_file_length)
    audio_file = np.zeros(int(audio_duration * sr))
    audio_file_length = len(audio_file)

    file_timestamps_start = []
    file_timestamps_end = []
    speakers = []

    start = 0

    for element in batch:
        audio_segment = element["audio"]["array"]
        dur = len(audio_segment) / sr
        end = start + dur

        file_timestamps_start.append(start)
        file_timestamps_end.append(end)
        speakers.append(element["client_id"])

        start_index = int(start * sr)

        if start_index >= audio_file_length:
            break

        segment_length = min(audio_file_length - start_index, len(audio_segment))

        audio_file[start_index : start_index + segment_length] += audio_segment[
            :segment_length
        ]

        start = max(int(0), np.random.normal(end, std_concatenate))

    audio_file = {
        "array": audio_file,
        "sampling_rate": sr,
    }

    new_batch["speakers"].append(speakers)
    new_batch["audio"].append(audio_file)
    new_batch["timestamps_start"].append(file_timestamps_start)
    new_batch["timestamps_end"].append(file_timestamps_end)

    return new_batch


dataset = common_voice["train"].shuffle()

dataset = dataset.select(range(200))

dataset = dataset.select_columns(["client_id", "audio"])

audio_file_length = 1.1
std_concatenate = 3

dataset = dataset.map(
    lambda example: concatenate_no_timestamps(
        example, audio_file_length, std_concatenate
    ),
    batched=True,
    batch_size=16,
    remove_columns=dataset.column_names,
    num_proc=24,
)

dataset.push_to_hub("kamilakesbi/commonvoice_en_spd_train_small_test")
