import numpy as np
import torch


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

        batch["waveforms"] = torch.stack([f["waveforms"] for f in features])

        return batch
