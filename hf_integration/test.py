import pytorch_lightning as pl
from pyannote.database import registry

from pyannote.audio.models.segmentation import PyanNet
from pyannote.audio.tasks import SpeakerDiarization

registry.load_database(
    "/home/kamil/projects/AMI-diarization-setup/pyannote/database.yml"
)
ami = registry.get_protocol("AMI.SpeakerDiarization.mini")

vad_task = SpeakerDiarization(ami, duration=2.0, batch_size=128)

vad_model = PyanNet(task=vad_task, sincnet={"stride": 10})

trainer = pl.Trainer(devices=1, max_epochs=1)
trainer.fit(vad_model)
