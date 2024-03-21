import numpy as np
import torch
import torchaudio.transforms as T
from audiomentations import AddBackgroundNoise, ApplyImpulseResponse, Compose
from datasets import Dataset, DatasetDict, load_dataset
from denoiser import pretrained
from denoiser.dsp import convert_audio


class ASR_to_SPD_dataset:
    def __init__(
        self,
        config,
    ):
        self.audio_file_length = config["audio_file_length"]
        self.std_concatenate = config["std_concatenate"]
        self.sample_rate = config["sample_rate"]
        self.refine_with_vad = config["refine_with_vad"]
        self.denoise = config["denoise"]
        self.normalize = config["normalize"]
        self.augment = config["augment"]

        if self.denoise:
            self.denoiser = pretrained.dns64().cuda()

        if self.refine_with_vad:

            torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
            vad_model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=True
            )
            self.vad_model = vad_model
            self.get_speech_timestamps = utils[0]

        if self.augment:

            self.bn_path = config["bn_path"]
            self.ir_path = config["ir_path"]

            self.augmentation_pipeline = Compose(
                [
                    ApplyImpulseResponse(self.ir_path, p=0.8),
                    AddBackgroundNoise(self.bn_path, 10, 15, p=0.8),
                ]
            )

    def estimate_audio_duration(self, batch, sr):

        audio_duration = 0
        for row in batch:
            audio_duration += len(row["audio"]["array"]) / sr

        audio_duration *= self.audio_file_length

        return audio_duration

    def normalize_audio(self, audio_segment):

        return audio_segment / max(np.max(audio_segment), -np.min(audio_segment))

    def denoise_audio(self, audio_file):

        audio_file_converted = convert_audio(
            torch.tensor(audio_file).unsqueeze(0).cuda(),
            self.sample_rate,
            self.denoiser.sample_rate,
            self.denoiser.chin,
        )
        with torch.no_grad():
            audio_file = (
                self.denoiser(torch.tensor(audio_file_converted, dtype=torch.float32))[
                    0
                ]
                .squeeze(0)
                .cpu()
                .numpy()
            )

        return audio_file

    def augment_audio(self, audio_file):

        audio_file = self.augmentation_pipeline(
            samples=audio_file, sample_rate=self.sample_rate
        )
        return audio_file

    def refine_timestamps(self, audio_segment, speaker, start):

        speech_timestamps = self.get_speech_timestamps(
            audio_segment, self.vad_model, sampling_rate=self.sample_rate
        )

        file_timestamps_start = [
            start + timestamps["start"] / self.sample_rate
            for timestamps in speech_timestamps
        ]
        file_timestamps_end = [
            start + timestamps["end"] / self.sample_rate
            for timestamps in speech_timestamps
        ]
        speakers = [speaker] * len(speech_timestamps)

        return (file_timestamps_start, file_timestamps_end, speakers)

    def concatenate_no_timestamps(
        self,
        files,
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

        audio_duration = self.estimate_audio_duration(batch, sr)
        audio_file = np.zeros(int(audio_duration * self.sample_rate))
        audio_file_length = len(audio_file)

        start = 0

        file_timestamps_start = []
        file_timestamps_end = []
        speakers = []

        for element in batch:

            audio_segment = element["audio"]["array"]

            if self.sample_rate:
                resample = T.Resample(sr, self.sample_rate)
                audio_segment = (
                    resample(torch.tensor(audio_segment, dtype=torch.float32))
                    .cpu()
                    .numpy()
                )

            if self.normalize:
                audio_segment = self.normalize_audio(audio_segment)

            dur = len(audio_segment) / self.sample_rate
            end = start + dur

            start_index = int(start * self.sample_rate)

            if start_index >= audio_file_length:
                break

            segment_length = min(audio_file_length - start_index, len(audio_segment))

            if self.refine_with_vad:
                (
                    file_timestamps_start_vad,
                    file_timestamps_end_vad,
                    speakers_vad,
                ) = self.refine_timestamps(
                    audio_segment,
                    element["client_id"],
                    start,
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
            start = max(int(0), np.random.normal(end, self.std_concatenate))

        if self.denoise:
            audio_file = self.denoise_audio(audio_file)

        if self.augment:
            audio_file = self.augment_audio(audio_file)

        if self.normalize:
            audio_file = self.normalize_audio(audio_file)

        audio_file = {
            "array": np.array(audio_file),
            "sampling_rate": self.sample_rate,
        }

        new_batch["speakers"].append(speakers)
        new_batch["audio"].append(audio_file)
        new_batch["timestamps_start"].append(file_timestamps_start)
        new_batch["timestamps_end"].append(file_timestamps_end)

        return new_batch


def create_spd_dataset_from_asr(
    asr_dataset,
    speaker_column_name,
    audio_column_name,
    config,
    batch_size,
    num_proc=12,
):

    subsets = ["train", "validation", "test"]

    speaker_diarization_dataset = DatasetDict(
        {
            "train": Dataset.from_dict({}),
            "validation": Dataset.from_dict({}),
            "test": Dataset.from_dict({}),
        }
    )

    asr_dataset.select_columns([str(speaker_column_name), str(audio_column_name)])

    asr_to_spd = ASR_to_SPD_dataset(config)

    for subset in subsets:

        dataset = asr_dataset[str(subset)].shuffle()
        dataset = dataset.select(range(200))

        dataset = dataset.map(
            lambda example: asr_to_spd.concatenate_no_timestamps(example),
            batched=True,
            batch_size=batch_size,
            remove_columns=dataset.column_names,
            num_proc=num_proc,
        )

        speaker_diarization_dataset[str(subset)] = dataset

    return speaker_diarization_dataset


if __name__ == "__main__":

    config = {
        "audio_file_length": 1.1,
        "std_concatenate": 2,
        "sample_rate": 16000,
        "refine_with_vad": True,
        "denoise": True,
        "normalize": True,
        "augment": True,
        "bn_path": "/home/kamil/datasets/wham_noise/wham_noise/tr",
        "ir_path": "/home/kamil/datasets/MIT-ir-survey",
    }

    batch_size = 8
    num_proc = 1

    common_voice = load_dataset(
        "mozilla-foundation/common_voice_16_1", "en", num_proc=2
    )
    speaker_column_name = "client_id"
    audio_column_name = "audio"

    spd_dataset = create_spd_dataset_from_asr(
        common_voice,
        speaker_column_name,
        audio_column_name,
        config,
        batch_size,
        num_proc,
    )

    spd_dataset.push_to_hub("kamilakesbi/commonvoice_en_spd_train_small_test")
