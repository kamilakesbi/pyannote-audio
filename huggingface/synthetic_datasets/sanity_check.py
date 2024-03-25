import numpy as np
from datasets import load_dataset
from pyannote.database import registry

from pyannote.audio import Model
from pyannote.audio.tasks import SpeakerDiarization


def get_chunk_from_pyannote(seg_task, file_id, start_time, duration):

    seg_task.prepare_data()
    seg_task.setup()

    chunk = seg_task.prepare_chunk(file_id, start_time, duration)

    return chunk


if __name__ == "__main__":

    registry.load_database(
        "/home/kamil/projects/AMI-diarization-setup/pyannote/database.yml"
    )
    ami = registry.get_protocol("AMI.SpeakerDiarization.only_words")

    seg_task = SpeakerDiarization(
        ami, duration=10.0, max_speakers_per_chunk=3, max_speakers_per_frame=2
    )
    pretrained = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token=True)
    seg_task.model = pretrained

    synthetic_ami_dataset_processed = load_dataset(
        "kamilakesbi/ami_spd_nobatch_processed_sc"
    )

    # Extract 10 second audio from meeting EN2001a (= file_id 124).
    # We choose start_time = 3.34 to match with the first 10 seconds of audio of the synthetic AMI.
    real_ami_chunk = get_chunk_from_pyannote(seg_task, 124, 3.34, 10)

    synthetic_ami_chunk = synthetic_ami_dataset_processed["train"][0]

    real_labels = real_ami_chunk["y"].data
    synthetic_labels = np.array(synthetic_ami_chunk["labels"])

    assert (synthetic_labels == real_labels).all(), "labels are not matching"
