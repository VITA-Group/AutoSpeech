
## Mel-filterbank
window_length = 25  # In milliseconds
window_step = 10    # In milliseconds
n_fft = 512


## Audio
sampling_rate = 16000
# Number of spectrogram frames in a partial utterance
partials_n_frames = 160     # 1600 ms
# Number of spectrogram frames at inference
inference_n_frames = 80     #  800 ms


## Audio volume normalization
audio_norm_target_dBFS = -30

