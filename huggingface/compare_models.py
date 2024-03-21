from metrics import test
from pyannote.database import registry
from segmentation.pretrained_model import SegmentationModel, SegmentationModelConfig

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

print("Original Pretrained: ")

test_file = next(ami.test())

spk_probability1 = Inference(pretrained, step=2.5)(test_file)

der_pretrained = test(model=pretrained, protocol=ami, subset="test")
print(f"Local DER (pretrained) = {der_pretrained}%")

print("Fine-tuned model: ")

config = SegmentationModelConfig()
model = SegmentationModel(config).from_pretrained(
    "/home/kamil/projects/pyannote-audio/output/checkpoint-6432"
)
model = model.to_pyannote_model()

spk_probability = Inference(model, step=2.5)(test_file)

der_pretrained = test(model=model, protocol=ami, subset="test")
print(f"Local DER (fine-tuned) = {der_pretrained}%")
