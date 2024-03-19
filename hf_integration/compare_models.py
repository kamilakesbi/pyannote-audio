from pyannote.database import registry
from segmentation_model.pretrained_model import (
    SegmentationModel,
    SegmentationModelConfig,
)

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

print("Original Pretrained: ")

test_file = next(ami.test())

spk_probability = Inference(pretrained, step=2.5)(test_file)

der_pretrained = test(model=pretrained, protocol=ami, subset="test")
print(f"Local DER (pretrained) = {der_pretrained * 100:.1f}%")

print("Fine-tuned model: ")

config = SegmentationModelConfig()
model = SegmentationModel(config).from_pretrained(
    "/home/kamil/projects/pyannote-audio/output/checkpoint-1233"
)
model = model.to_pyannote_model()

spk_probability = Inference(model, step=2.5)(test_file)

der_pretrained = test(model=model, protocol=ami, subset="test")
print(f"Local DER (fine-tuned) = {der_pretrained * 100:.1f}%")
