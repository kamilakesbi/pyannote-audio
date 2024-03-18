import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


def pad_targets(labels, speakers, max_speakers_per_chunk=3):

    targets = []

    for i in range(len(labels)):

        label = speakers[i]
        target = labels[i].numpy()
        num_speakers = len(label)

        if num_speakers > max_speakers_per_chunk:
            indices = np.argsort(-np.sum(target, axis=0), axis=0)
            target = target[:, indices[:max_speakers_per_chunk]]

        elif num_speakers < max_speakers_per_chunk:
            target = np.pad(
                target,
                ((0, 0), (0, max_speakers_per_chunk - num_speakers)),
                mode="constant",
            )

        targets.append(target)

    return torch.from_numpy(np.stack(targets))


class DataCollator:
    def __call__(self, features):
        batch = {}

        speakers = [f["nb_speakers"] for f in features]
        labels = [f["labels"] for f in features]

        batch["labels"] = pad_targets(labels, speakers)
        # batch["labels"] = pad_sequence(
        #     [f["labels"] for f in features], batch_first=True, padding_value=0
        # )
        batch["waveforms"] = pad_sequence(
            [f["waveforms"] for f in features], batch_first=True, padding_value=0
        )
        return batch
