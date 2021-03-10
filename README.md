# Automatic Polyphonic Piano Transcription with Recurrent Neural Networks
IPython-notebook templates for neural network training and then using it to generate piano MIDI-files from audio (MP3, WAV, etc.).  The accuracy will depend on the complexity of the song, and will obviously be higher for solo piano pieces.

# Update (2021 March)

There is another pre-trained __*Magenta's*__ model in __*TensorFlow Lite*__ format, it can be downloaded here:
https://storage.googleapis.com/magentadata/models/onsets_frames_transcription/tflite/onsets_frames_wavinput.tflite
or look for the link in the GitHub-repository:
https://github.com/magenta/magenta/tree/master/magenta/models/onsets_frames_transcription/realtime

It takes as input approximately 1 second of raw audio (not 20 seconds of mel spectrogram).  There is an example of using the model in my fifth IPython template ("5 TF Lite Inference.ipynb").  This model is super-fast on my Android device and accuracy is still not bad.  To see my app for Android 8.0 (API level 26) or higher, click on the following screenshot:

[![](https://raw.GitHubUserContent.com/BShakhovsky/BShakhovsky.github.io/master/Android.png 'Android 8.0')](https://GitHub.com/BShakhovsky/PianoTranscription_Android/blob/master/README.md)

or get it on Google Play:

[![](https://play.google.com/intl/en_us/badges/static/images/badges/en_badge_web_generic.png 'Get it on Google Play')](https://play.google.com/store/apps/details?id=ru.BShakhovsky.Piano_Transcription)

The previous full Tensorflow model (not TensorFlow Lite) is used in my app for Windows 7 or later, to see it click on the following screenshot:

[![](https://GitHub.com/BShakhovsky/PianoTranscription3D/raw/master/Keyboard.png 'Windows 7')](https://GitHub.com/BShakhovsky/PianoTranscription3D/blob/master/README.md)


# Update (2019 June)

There is Google's model called __*"Onsets & Frames"*__ with very good accuracy, see the following blog post:
https://magenta.tensorflow.org/onsets-frames
or GitHub-repository:
https://github.com/magenta/magenta/tree/master/magenta/models/onsets_frames_transcription

Just for fun, I blindly copied those model parameters, and trained the model in the second IPython template.  But my resultant accuracy was slightly less, probably because of the reduced batch size.  So, eventually, in the third IPython template, instead of training my own model, I just copied the weights from the Google's pre-trained tensorflow checkpoint:
https://storage.googleapis.com/magentadata/models/onsets_frames_transcription/maestro_checkpoint.zip

# Troubleshooting
If Python __*"Librosa"*__ module cannot open any audio format except WAV, __*download FFmpeg codec:*__

[![](https://upload.wikimedia.org/wikipedia/commons/thumb/5/5f/FFmpeg_Logo_new.svg/500px-FFmpeg_Logo_new.svg.png 'FFmpeg')](https://web.archive.org/web/20200918014242/https://ffmpeg.zeranoe.com/builds/)

Choose your Windows architecture and static linking there.  And do not forget to add the __*"PATH"*__ environment variable with the location of your downloaded __*"ffmpeg.exe"*__.  Or see more detailed instructions here:

https://www.wikihow.com/Install-FFmpeg-on-Windows


# Dataset: MAESTRO (MIDI and Audio Edited<br>for Synchronous TRacks and Organization)
Downloaded from Google Magenta: https://magenta.tensorflow.org/datasets/maestro#download
## Warning
For some samples last midi note onsets are slightly beyond the duration of the corresponding WAV-audio.

# Not used datasets
## Not used: MAPS (from Fichiers - Aix-Marseille Université)
https://amubox.univ-amu.fr/index.php/s/iNG0xc5Td1Nv4rR

### Issue 1 (small dataset and not as natural)
From https://arxiv.org/pdf/1810.12247.pdf, Page 4, Section 3 "Dataset":

`MAPS ... “performances” are not as natural as the MAESTRO performances captured from live performances.  In addition, synthesized audio makes up a large fraction of the MAPS dataset.`

### Issue 2 (skipped notes)
From https://arxiv.org/pdf/1710.11153.pdf, Page 6, Section 6 "Need for more data, more rigorous evaluation":

`In addition to the small number of the MAPS Disklavier recordings, we have also noticed several cases where the Disklavier appears to skip some notes played at low velocity. For example, at the beginning of the Beethoven Sonata No. 9, 2nd movement, several Ab notes played with MIDI velocities in the mid-20s are clearly missing from the audio...`

### Issue 3 (two chords instead of one)
There is an issue with datasets __*"ENSTDkAm"*__ & __*"ENSTDkCl"*__, subtypes __*"RAND"*__ & __*"UCHO"*__.  They are assumed to have only one chord per one WAV-file.  But sometimes the chord is split into two onset times in corresponding MIDI and TXT-files, and those two onset times fall into two consecutive time-frames of cqt-transform (or mel-transform).
## Not used: MusicNET (from University of Washington Computer Science & Engineering)
https://homes.cs.washington.edu/~thickstn/musicnet.html

From https://arxiv.org/pdf/1810.12247.pdf, Page 4, Section 3 "Dataset":

`As discussed in Hawthorne et al. (2018), the alignment between audio and score is not fully accurate.  One advantage of MusicNet is that it contains instruments other than piano` ... `and a wider variety of recording environments.`


# 1 Datasets Preparation
## Train/Test Split
From https://arxiv.org/pdf/1810.12247.pdf, Page 4, Section 3.2 "Dataset Splitting":
1. `No composition should appear in more than one split.`
2. `... proportions should be true globally and also within each composer.  Maintaining these proportions is not always possible because some composers have too few compositions in the dataset.`
3. `The validation and test splits should contain a variety of compositions.  Extremely popular compositions performed by many performers should be placed in the training split.`
4. `... we recommend using the splits which we have provided.`

## Mel-transform parameters
Don't know why not use Constant-Q transform, but from https://arxiv.org/pdf/1710.11153.pdf, Page 2, Section 3 "Model Configuration":

`We use librosa` ... `to compute the same input data representation of mel-scaled spectrograms with log amplitude of the input raw audio with 229 logarithmically-spaced frequency bins, a hop length of 512, an FFT window of 2048, and a sample rate of 16kHz.`

From https://arxiv.org/pdf/1810.12247.pdf, Page 5, Section 4 "Piano Transcription":

`switched to HTK frequency spacing (Young et al., 2006) for the mel-frequency spectrogram input.`

Mel-frequency values are strange:
- fmin = 30 Hz, but the first "A" note of 1st octave is *27.5 Hz*
- fmax = 8 000 Hz (librosa default), and it is much higher than the last "C" note of 8th octave (*4 186 Hz*).  So, mel-spectrogram will contain lots of high harmonics, and maybe, it will help the CNN-model correctly identify notes in the last octaves.

Maybe (don't know) Mel-scaled spectrogram is used instead of Constant-Q transform, because CQT-transform produces equal number of bins for each note, while mel-frequencies are located such that there are more nearby frequencies for higher notes.  So, mel-spectrogram provides more input data for higher octaves, and the CNN-model can transcribe higher notes with better accuracy.  It can help solve the issue with lots of annoying false-positive notes in high octaves.

## Additional non-linear logarithmic scaling
librosa.power_to_db, ref=1 (default) --> mels decibels are approximately in range *[-40 ... +40]*

## Note message durations: 2 consecutive frames
From https://arxiv.org/pdf/1710.11153.pdf:

Page 2, Section 2 "Dataset and Metrics":

`... we first translate “sustain pedal” control changes into longer note durations.  If a note is active when sustain goes on, that note will be extended until either sustain goes off or the same note is played again.`

Page 3, Section 3, "Model Configuration":

`... all onsets will end up spanning exactly two frames.  Labeling only the frame that contains the exact beginning of the onset does not work as well because of possible mis-alignments of the audio and labels.  We experimented with requiring a minimum amount of time a note had to be present in a frame before it was labeled, but found that the optimum value was to include any presence.`

## Number of time-frames: 625 + 1 (20 seconds at sample rate of 16 kHz)
From https://arxiv.org/pdf/1710.11153.pdf, Page 2, Section 3 "Model Configuration":

`... we split the training audio into smaller files.  However, when we do this splitting we do not want to cut the audio during notes because the onset detector would miss an onset while the frame detector would still need to predict the note’s presence.  We found that 20 second splits allowed us to achieve a reasonable batch size during training of at least 8, while also forcing splits in only a small number of places where notes are active.`


# 2 Training and Validation
My previous model performed well on __*MAPS*__ dataset, but resulted in much lower accuracy on new larger, more natural, and more complicated __*MAESTRO*__ dataset.  It turned out, that just simple fully-connected network produced similar result.  It probably makes sense, as it is written in https://arxiv.org/pdf/1810.12247.pdf, Page 5, Section 4 "Piano Transcription":

`In general, we found that the best ways to get higher performance with the larger dataset were to make the model larger and simpler.`

So, I based my model on Google "Onsets and Frames: Dual-Objective Piano Transcription".<br>From https://arxiv.org/pdf/1710.11153.pdf, Page 3, Figure 1 "Diagram of Network Architecture":

![](https://magenta.tensorflow.org/assets/onsets_frames/networkstack9.png 'Architecture')

I blindly copied those model parameters, except:
1. Batch normalization is used wherever possible (everywhere except LSTM layer and the last fully-connected layer).
2. Dropout is not required at all, because there is no sign of overfitting.
3. It was impossible to keep the recommended batch size of 8 on my machine, so I reduced it to 4.

## Onsets CNN subnetwork
From https://arxiv.org/pdf/1710.11153.pdf, Page 2, Section 3 "Model Configuration":

`The onset detector is composed of the acoustic model, followed by a bidirectional LSTM ... with 128 units in both the forward and backward directions, followed by a fully connected sigmoid layer with 88 outputs for representing the probability of an onset for each of the 88 piano keys.`

Acoustic convolutional stack is taken from https://arxiv.org/pdf/1612.05153.pdf, Page 5, Section 5 "State of the Art Models", Table 4 "Model Architectures", ConvNet column.  In our case there are 625 + 1 frames (20 seconds), there is bidirectional __*LSTM*__ layer in between the last two fully-connected layers, batch normalization is heavily used, dropout is not required, and remaining parameters are from https://arxiv.org/pdf/1810.12247.pdf (also, from https://github.com/magenta/magenta/blob/master/magenta/models/onsets_frames_transcription/model.py ):

Page 5, Section 4 "Piano Transcription":
1. `... increased the size of the bidirectional LSTM layers from 128 to 256 units,`
2. `changed the number of filters in the convolutional layers from 32/32/64 to 48/48/96,`
3. `and increased the units in the fully connected layer from 512 to 768.`

## Offsets RNN subnetwork
From https://arxiv.org/pdf/1810.12247.pdf, Page 5, Section 4 "Piano Transcription":

`... adding an offset detection head, inspired by Kelz et al. (2018).  The offset head feeds into the frame detector but is not directly used during decoding.  The offset labels are defined to be the 32ms following the end of each note.`

## Frames RNN combined network
From https://arxiv.org/pdf/1710.11153.pdf, Page 2, Section 3 "Model Configuration":

`The frame activation detector is composed of a separate acoustic model, followed by a fully connected sigmoid layer with 88 outputs.  Its output is concatenated together with the output of the onset detector and followed by a bidirectional LSTM with 128 units in both the forward and backward directions.  Finally, the output of that LSTM is followed by a fully connected sigmoid layer with 88 outputs.  During inference, we use a threshold of 0.5 to determine whether the onset detector or frame detector is active.`

From https://arxiv.org/pdf/1810.12247.pdf, Page 5, Section 4 "Piano Transcription":

`We also stopped gradient propagation into the onset subnetwork from the frame network...`

## Frame-Based Accuracy Metric (stricter than F1-score)
Standard accuracy is not representative, it would be almost 50% immediately at the beginning, and would obviously rise up to 99% very quickly.  F1-score is better, but there is stricter accuracy metric, taken from<br>http://c4dm.eecs.qmul.ac.uk/ismir15-amt-tutorial/AMT_tutorial_ISMIR_2015.pdf<br>Page 50, Slide 100, Evaluation Metric 3 [Dixon, 2000]:

Acc = tp / (fp + fn + tp) = tp / ((pp - tp) + (rp - tp) + tp) = tp / (pp + rp - tp)

## Volumes CNN - separate network
From https://arxiv.org/pdf/1710.11153.pdf, Page 3, Section 3.1 "Velocity Estimation":

`We further extend the model by adding another stack to also predict velocities for each onset.  This stack is similar to the others and consists of the same layers of convolutions.  This stack does not connect to the other two.  The velocity labels are generated by dividing all the velocities by the maximum velocity present in the piece.  The smallest velocity does not go to zero, but rather to Vmin ... Vmax.`

`At inference time the output is clipped to [0, 1] and then transformed to a midi velocity by the following mapping:`

Vmidi = 80 * Vpredicted + 10

## 2.1. Training & Validation
From https://arxiv.org/pdf/1710.11153.pdf (also from https://github.com/magenta/magenta/blob/master/magenta/models/onsets_frames_transcription/model.py ):

Page 4, Section 4 "Experiments":

`... batch size of 8 ...` __*In our case it is 4 for onsets, offsets, and frames subnetworks, and 8 for volumes CNN*__

`... Adam optimizer ...`

### Keras Backend
We use __*CuDNNLSTM*__ as recurrent layer, and it supports only __*Tensorflow*__ backend, so, it is impossible to use __*CNTK*__.

## 2.2 Testing
From https://arxiv.org/pdf/1710.11153.pdf, Page 2, Section 2 "Dataset and Metrics":
1. `We use the mir eval library ... to calculate notebased precision, recall, and F1 scores.`
2. `... we calculate two versions of note metrics: one requiring that onsets be within ±50ms of ground truth but ignoring offsets and one that also requires offsets resulting in note durations within 20% of the ground truth or within 50ms, whichever is greater.`
3. `Both frame and note scores are calculated per piece and the mean of these per-piece scores is presented as the final metric for a given collection of pieces.`
4. `Poor quality transcriptions can still result in high frame scores due to short spurious notes and repeated notes that should be held.`

Page 4, Section 4 "Experiments":

`To our ears, the decrease in transcription quality is best reflected by the note-with-offset scores.`

From https://arxiv.org/pdf/1810.12247.pdf, Page 5, Section 4 "Piano Transcription", Table 4:

`Note-based scores calculated by the mir eval library, frame-based scores as defined in Bay et al. (2009). Final metric is the mean of scores calculated per piece.`


# 3 Magenta to Keras

In the third IPython template, instead of training my own model, I just copied the weights from the Google's pre-trained tensorflow checkpoint:
https://storage.googleapis.com/magentadata/models/onsets_frames_transcription/maestro_checkpoint.zip

Or look for the link here:
https://github.com/magenta/magenta/tree/master/magenta/models/onsets_frames_transcription


# 4 Piano Audio to Midi

No instrument information is extracted, and all transcribed notes get combined into one part.

From https://arxiv.org/pdf/1710.11153.pdf:

Page 1 "Abstract":

`During inference, we restrict the predictions from the framewise detector by not allowing a new note to start unless the onset detector also agrees that an onset for that pitch is present in the frame.`

Page 2, Section 3 "Model Configuration":

`An activation from the frame detector is only allowed to start a note if the onset detector agrees that an onset is present in that frame.`

There could be lots of noticeable false-positive notes in high octaves where they absolutely surely should not be.  If it is too annoying, then I cannot think of any better solution than to use much higher threshold for higher octaves :disappointed:

# 5 TF Lite Inference

In the fifth IPython template I used another pre-trained __*Magenta's*__ model in __*TensorFlow Lite*__ format, it can be downloaded here:
https://storage.googleapis.com/magentadata/models/onsets_frames_transcription/tflite/onsets_frames_wavinput.tflite
or look for the link in the GitHub-repository:
https://github.com/magenta/magenta/tree/master/magenta/models/onsets_frames_transcription/realtime

It takes as input approximately 1 second of raw audio (not 20 seconds of mel spectrogram).  This model is super-fast on my Android device and accuracy is still not bad.