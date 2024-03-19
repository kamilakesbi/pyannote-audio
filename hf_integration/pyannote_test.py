import argparse
from copy import deepcopy

import pytorch_lightning as pl
from pyannote.database import registry

from pyannote.audio import Inference, Model
from pyannote.audio.tasks import SpeakerDiarization


def test(model, protocol, subset="test"):
    from pyannote.audio.pipelines.utils import get_devices
    from pyannote.audio.utils.metric import DiscreteDiarizationErrorRate
    from pyannote.audio.utils.signal import binarize

    (device,) = get_devices(needs=1)
    metric = DiscreteDiarizationErrorRate()
    files = list(getattr(protocol, subset)())

    inference = Inference(model, device=device)

    for file in files:
        reference = file["annotation"]
        hypothesis = binarize(inference(file))
        uem = file["annotated"]
        _ = metric(reference, hypothesis, uem=uem)

    return abs(metric)


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
