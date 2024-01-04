from multiprocessing import set_start_method
set_start_method('fork', force=True)

import librosa
import torch
import time

t1 = time.time()
pipeline = torch.hub.load('pyannote/pyannote-audio', 'dia', device="cpu")
t2 = time.time()
print(("pipeline load execution time: {0}".format(t2-t1)))


def diarize(path):
    """ Diarize the given sound file

    :param path: path of the sound file
    :return:
    """
    # apply diarization pipeline on your audio file
    t1 = time.time()
    diarization = pipeline({'audio': path})
    t2 = time.time()
    print("diarization pipeline execution time: {0}".format(t2-t1))

    response = {}
    response["speakers"] = []

    y, sr = librosa.load(path)
    sound_duration = librosa.get_duration(y, sr)

    # iterate over speech turns
    t1 = time.time()
    for turn, aa, predicted_speaker in diarization.itertracks(yield_label=True):
        predicted_speaker = "Speaker - " + predicted_speaker
        start_percentage = (turn.start / sound_duration) * 100
        end_percentage = (turn.end / sound_duration) * 100

        found = False
        for speaker in response["speakers"]:
            if speaker["name"] == "Speaker - " + predicted_speaker:
                found = True
                speaker["time_intervals"].append([turn.start,turn.end])
                speaker["time_percentages"].append([start_percentage, end_percentage])
        if not found:
            response["speakers"].append({"name": predicted_speaker,
                                         "time_intervals": [[turn.start, turn.end]],
                                         "time_percentages": [[start_percentage, end_percentage]]})

        print("Speaker {0} speaks between t={1}s and t={2}s.".format(predicted_speaker, turn.start,turn.end))

    t2 = time.time()
    print("diarization iteration execution time: {0}".format(t2-t1))
    return response


if __name__ == "__main__":
    diarize("/Users/zeynep/Desktop/diarizationdeneme/on_crypto.wav")
