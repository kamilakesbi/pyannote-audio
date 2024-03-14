import numpy as np

from pyannote.audio.utils.permutation import permutate


def discrete_diarization_error_rate(reference: np.ndarray, hypothesis: np.ndarray):
    """Discrete diarization error rate

    Parameters
    ----------
    reference : (num_frames, num_speakers) np.ndarray
        Discretized reference diarization.
        reference[f, s] = 1 if sth speaker is active at frame f, 0 otherwise
    hypothesis : (num_frames, num_speakers) np.ndarray
        Discretized hypothesized diarization.
       hypothesis[f, s] = 1 if sth speaker is active at frame f, 0 otherwise

    Returns
    -------
    der : float
        (false_alarm + missed_detection + confusion) / total
    components : dict
        Diarization error rate components, in number of frames.
        Keys are "false alarm", "missed detection", "confusion", and "total".
    """

    reference = reference.astype(np.half)
    hypothesis = hypothesis.astype(np.half)

    # permutate hypothesis to maximize similarity to reference
    (hypothesis,), _ = permutate(reference[np.newaxis], hypothesis)

    # total speech duration (in number of frames)
    total = 1.0 * np.sum(reference)

    # false alarm and missed detection (in number of frames)
    detection_error = np.sum(hypothesis, axis=1) - np.sum(reference, axis=1)
    false_alarm = np.maximum(0, detection_error)
    missed_detection = np.maximum(0, -detection_error)

    # speaker confusion (in number of frames)
    confusion = np.sum((hypothesis != reference) * hypothesis, axis=1) - false_alarm

    false_alarm = np.sum(false_alarm)
    missed_detection = np.sum(missed_detection)
    confusion = np.sum(confusion)

    if total != 0:
        return (false_alarm + missed_detection + confusion) / total
    else:
        return 0


def der_metric(eval_pred):

    logits, labels = eval_pred
    predictions = (logits >= 0.5).astype(np.float32)

    metric = 0
    for i in range(len(predictions)):
        prediction = predictions[i]
        label = labels[i]

        metric += discrete_diarization_error_rate(label, prediction)

    metric /= len(predictions)

    return {"der": metric}
