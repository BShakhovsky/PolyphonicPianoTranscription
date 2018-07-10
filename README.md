# Automatic Polyphonic Piano Transcription with Convolutional Neural Networks

IPython-notebook templates for generating piano MIDI-files from audio (MP3, WAV, etc.).  The accuracy depends on the complexity of the song, and is obviously higher for solo piano pieces.


# Troubleshooting

If librosa cannot open any audio format except WAV, __*download FFmpeg codec:*__

[![](https://upload.wikimedia.org/wikipedia/commons/thumb/5/5f/FFmpeg_Logo_new.svg/1280px-FFmpeg_Logo_new.svg.png 'FFmpeg')](https://ffmpeg.org)

And do not forget to add the __*"PATH"*__ environment variable with the location of your downloaded __*"ffmpeg.exe"*__.


# Datasets

### MAPS (MIDI Aligned Piano Sounds)

https://amubox.univ-amu.fr/index.php/s/iNG0xc5Td1Nv4rR

### International Audio Laboratories Erlangen

https://www.audiolabs-erlangen.de/resources/MIR/SMD/midi


# 1 Datasets Preparation

Constant-Q transform with the following parameters to play around:

1. Number of frequency bins per note (now it is 5)
2. Number of time-frames before and after of note onset time to provide some context (now it is 2 before and 2 after).
3. Margin for harmonic-percussive separation (now it is 4).

__*TODO:*__ try non-linear log scaling of the cqt amplitude (librosa.amplitude_to_db).


# 2 Training and Validation

The percentage of samples in the evaluation split (25% of data not used for training) where every single one of the 88 rounded outputs was correct -- that came out to around 59%.  F1-score would be higher.


# 3 Piano Audio to Midi

No instrument information is extracted, and all transcibed notes get combined into one part.