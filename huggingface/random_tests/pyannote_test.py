import argparse
from copy import deepcopy

import pytorch_lightning as pl
from metrics import test
from pyannote.database import registry

from pyannote.audio import Inference, Model
from pyannote.audio.tasks import SpeakerDiarization

registry.load_database(
    "/home/kamil/projects/AMI-diarization-setup/pyannote/database.yml"
)
ami = registry.get_protocol("AMI.SpeakerDiarization.mini")

pretrained = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token=True)
seg_task = SpeakerDiarization(
    ami, duration=10.0, max_speakers_per_chunk=3, max_speakers_per_frame=2
)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--action", help="", default="train")

    args = parser.parse_args()

    if str(args.action) == "train":

        finetuned = deepcopy(pretrained)
        finetuned.task = seg_task

        trainer = pl.Trainer(devices=1, max_epochs=1)
        trainer.fit(finetuned)

    elif str(args.action) == "test":

        test_file = next(ami.test())
        spk_probability = Inference(pretrained, step=2.5)(test_file)

        der_pretrained = test(model=pretrained, protocol=ami, subset="test")
        print(f"Local DER (pretrained) = {der_pretrained * 100:.1f}%")
