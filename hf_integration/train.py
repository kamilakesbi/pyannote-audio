from torch.nn.utils.rnn import pad_sequence


class DataCollator:
    def __call__(self, features):
        batch = {}

        batch["labels"] = pad_sequence(
            [f["labels"] for f in features], batch_first=True, padding_value=0
        )
        batch["waveforms"] = pad_sequence(
            [f["waveforms"] for f in features], batch_first=True, padding_value=0
        )

        return batch
