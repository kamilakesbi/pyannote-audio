import numpy as np
import torch
import torchaudio.transforms as T
from datasets import load_dataset

# 1. visit hf.co/pyannote/segmentation and accept user conditions
# 2. visit hf.co/settings/tokens to create an access token
# 3. instantiate pretrained voice activity detection pipeline


common_voice = load_dataset("mozilla-foundation/common_voice_16_1", "en", num_proc=24)


torch.hub.download_url_to_file(
    "https://models.silero.ai/vad_models/en.wav", "en_example.wav"
)


def estimate_audio_duration(batch, sr, audio_file_length=0.9):

    audio_duration = 0
    for row in batch:
        audio_duration += len(row["audio"]["array"]) / sr

    audio_duration *= audio_file_length

    return audio_duration


torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
model, utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=True
)

get_speech_timestamps = utils[0]


def refine_timestamps(audio_segment, sample_rate, speaker, start, end):

    speech_timestamps = get_speech_timestamps(
        audio_segment, model, sampling_rate=sample_rate
    )

    file_timestamps_start = [
        start + timestamps["start"] / 16000 for timestamps in speech_timestamps
    ]
    file_timestamps_end = [
        start + timestamps["end"] / 16000 for timestamps in speech_timestamps
    ]
    speakers = [speaker] * len(speech_timestamps)

    return (file_timestamps_start, file_timestamps_end, speakers)


def concatenate_no_timestamps(
    files,
    audio_file_length=1.1,
    std_concatenate=3,
    sample_rate=16000,
    refine_with_vad=True,
):

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
    audio_file = np.zeros(int(audio_duration * sample_rate))
    audio_file_length = len(audio_file)

    start = 0

    file_timestamps_start = []
    file_timestamps_end = []
    speakers = []

    for element in batch:

        audio_segment = element["audio"]["array"]

        if sample_rate:
            resample = T.Resample(sr, sample_rate)
            audio_segment = (
                resample(torch.tensor(audio_segment, dtype=torch.float32)).cpu().numpy()
            )

        dur = len(audio_segment) / sample_rate
        end = start + dur

        start_index = int(start * sample_rate)

        if start_index >= audio_file_length:
            break

        segment_length = min(audio_file_length - start_index, len(audio_segment))

        if refine_with_vad:
            (
                file_timestamps_start_vad,
                file_timestamps_end_vad,
                speakers_vad,
            ) = refine_timestamps(
                audio_segment,
                sample_rate,
                element["client_id"],
                start,
                end,
            )
            file_timestamps_start += file_timestamps_start_vad
            file_timestamps_end += file_timestamps_end_vad
            speakers += speakers_vad

        else:
            file_timestamps_start.append(start)
            file_timestamps_end.append(end)
            speakers.append(element["client_id"])

        audio_file[start_index : start_index + segment_length] += audio_segment[
            :segment_length
        ]
        start = max(int(0), np.random.normal(end, std_concatenate))

    audio_file = {
        "array": np.array(audio_file),
        "sampling_rate": sample_rate,
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
std_concatenate = 2
sample_rate = 16000
refine_with_vad = True

dataset = dataset.map(
    lambda example: concatenate_no_timestamps(
        example, audio_file_length, std_concatenate, sample_rate, refine_with_vad
    ),
    batched=True,
    batch_size=8,
    remove_columns=dataset.column_names,
    num_proc=1,
)

dataset.push_to_hub("kamilakesbi/commonvoice_en_spd_train_small_test")
