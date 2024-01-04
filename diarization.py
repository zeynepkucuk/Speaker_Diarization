
import torch
pipeline = torch.hub.load('pyannote/pyannote-audio', 'dia',device="cpu")
# apply diarization pipeline on your audio file
diarization = pipeline({'audio': '/Users/app/Downloads/zeynep_oğuzhan.wav'})
# iterate over speech turns
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f'Speaker "{speaker}" speaks between t={turn.start:.1f}s and t={turn.end:.1f}s.')


'''
Speaker "B" speaks between t=0.0s and t=0.5s.
Speaker "B" speaks between t=0.8s and t=10.0s.
Speaker "B" speaks between t=10.3s and t=15.4s.
Speaker "A" speaks between t=15.6s and t=16.6s.
Speaker "A" speaks between t=17.0s and t=23.0s.
Speaker "A" speaks between t=23.3s and t=25.8s.
Speaker "A" speaks between t=26.1s and t=28.0s.
Speaker "A" speaks between t=28.5s and t=31.4s.
Speaker "A" speaks between t=31.9s and t=33.2s.
Speaker "A" speaks between t=33.4s and t=34.1s.
Speaker "A" speaks between t=35.2s and t=37.7s.
Speaker "A" speaks between t=38.7s and t=41.9s.
Speaker "A" speaks between t=42.4s and t=48.6s.
Speaker "A" speaks between t=48.9s and t=50.0s.
Speaker "A" speaks between t=50.2s and t=51.7s.
Speaker "A" speaks between t=52.2s and t=56.5s.
Speaker "B" speaks between t=56.5s and t=60.6s.
Speaker "A" speaks between t=60.6s and t=67.9s.
Speaker "A" speaks between t=68.6s and t=69.5s.
Speaker "A" speaks between t=69.8s and t=70.2s.
Speaker "B" speaks between t=70.4s and t=72.9s.
Speaker "A" speaks between t=72.9s and t=73.5s.
Speaker "A" speaks between t=73.7s and t=76.4s.
Speaker "A" speaks between t=76.6s and t=79.1s.
Speaker "A" speaks between t=79.4s and t=80.6s.
Speaker "A" speaks between t=81.0s and t=83.3s.
Speaker "B" speaks between t=83.3s and t=90.3s.
Speaker "B" speaks between t=90.5s and t=94.2s.
Speaker "B" speaks between t=94.4s and t=98.7s.
Speaker "B" speaks between t=99.0s and t=104.8s.
Speaker "A" speaks between t=105.1s and t=106.8s.
Speaker "A" speaks between t=107.3s and t=110.4s.
Speaker "A" speaks between t=110.6s and t=110.8s.
Speaker "A" speaks between t=111.2s and t=114.3s.
Speaker "A" speaks between t=114.5s and t=115.8s.
Speaker "B" speaks between t=115.8s and t=118.7s.
Speaker "B" speaks between t=119.4s and t=134.0s.
Speaker "B" speaks between t=134.3s and t=135.8s.
Speaker "B" speaks between t=136.1s and t=137.3s.
Speaker "B" speaks between t=137.7s and t=141.6s.
Speaker "B" speaks between t=141.9s and t=145.0s.
'''
