from copy import deepcopy

import pytorch_lightning as pl
from pyannote.database import registry

from pyannote.audio import Model
from pyannote.audio.tasks import SpeakerDiarization

registry.load_database(
    "/home/kamil/projects/AMI-diarization-setup/pyannote/database.yml"
)
ami = registry.get_protocol("AMI.SpeakerDiarization.mini")

pretrained = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token=True)
seg_task = SpeakerDiarization(
    ami, duration=10.0, max_speakers_per_chunk=3, max_speakers_per_frame=2
)

finetuned = deepcopy(pretrained)
finetuned.task = seg_task

trainer = pl.Trainer(devices=1, max_epochs=1)
trainer.fit(finetuned)
