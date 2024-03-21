import numpy as np
import torch
import torchaudio.transforms as T
from datasets import load_dataset
from torchaudio import transforms

# from speechbrain.pretrained import SepformerSeparation as separator

common_voice = load_dataset("mozilla-foundation/common_voice_16_1", "en", num_proc=2)


# enhancer_model = separator.from_hparams(source="speechbrain/sepformer-wham16k-enhancement")

torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
vad_model, utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=True
)

get_speech_timestamps = utils[0]


def estimate_audio_duration(batch, sr, audio_file_length=0.9):

    audio_duration = 0
    for row in batch:
        audio_duration += len(row["audio"]["array"]) / sr

    audio_duration *= audio_file_length

    return audio_duration


def refine_timestamps(audio_segment, sample_rate, speaker, start, end):

    speech_timestamps = get_speech_timestamps(
        audio_segment, vad_model, sampling_rate=sample_rate
    )

    file_timestamps_start = [
        start + timestamps["start"] / sample_rate for timestamps in speech_timestamps
    ]
    file_timestamps_end = [
        start + timestamps["end"] / sample_rate for timestamps in speech_timestamps
    ]
    speakers = [speaker] * len(speech_timestamps)

    return (file_timestamps_start, file_timestamps_end, speakers)


def adjust_loudness(audio_segment, sample_rate, target_loudness):

    loudness = transforms.Loudness(sample_rate)
    estimated_loudness = (
        loudness(torch.tensor(audio_segment).unsqueeze(0)).cpu().numpy()
    )

    audio_segment = audio_segment * (10 ** (target_loudness - estimated_loudness / 20))

    return audio_segment


def concatenate_no_timestamps(
    files,
    audio_file_length=1.1,
    std_concatenate=3,
    sample_rate=16000,
    refine_with_vad=True,
    enhance=True,
    align_loudness=True,
    target_loudness=None,
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

        if align_loudness:
            sample_rate = sample_rate if sample_rate else sr
            adjust_loudness(audio_segment, sample_rate, target_loudness)

        # if enhance:
        #     audio_segment = enhancer_model(torch.tensor(audio_segment).unsqueeze(0)).squeeze(0,2).cpu().numpy()

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
enhance = False
align_loudness = True
target_loudness = -20


dataset = dataset.map(
    lambda example: concatenate_no_timestamps(
        example,
        audio_file_length,
        std_concatenate,
        sample_rate,
        refine_with_vad,
        enhance,
        align_loudness,
        target_loudness,
    ),
    batched=True,
    batch_size=8,
    remove_columns=dataset.column_names,
    num_proc=1,
)

dataset.push_to_hub("kamilakesbi/commonvoice_en_spd_train_small_test")
