# Automatic Polyphonic Piano Transcription with Convolutional Neural Networks

IPython-notebook templates for neural network training and then using it to generate piano MIDI-files from audio (MP3, WAV, etc.).  The accuracy will depend on the complexity of the song, and will obviously be higher for solo piano pieces.


# Troubleshooting

If Python __*"Librosa"*__ module cannot open any audio format except WAV, __*download FFmpeg codec:*__

[![](https://upload.wikimedia.org/wikipedia/commons/thumb/5/5f/FFmpeg_Logo_new.svg/500px-FFmpeg_Logo_new.svg.png 'FFmpeg')](https://ffmpeg.zeranoe.com/builds/)

Choose your Windows architecture and static linking there.  And do not forget to add the __*"PATH"*__ environment variable with the location of your downloaded __*"ffmpeg.exe"*__.  Or see more detailed instructions here:

https://www.wikihow.com/Install-FFmpeg-on-Windows


# Datasets

## MAPS (MIDI Aligned Piano Sounds)

Probably, it would be better to download it from here:

http://www.tsi.telecom-paristech.fr/aao/en/2010/07/08/maps-database-a-piano-database-for-multipitch-estimation-and-automatic-transcription-of-music/

But I was too lazy to find out how to sign up there, so I found another mirror here:

https://amubox.univ-amu.fr/index.php/s/iNG0xc5Td1Nv4rR

It is big and wonderful dataset, with several types of piano and recording room.  Thanks to it, I did not have to think how to generate my own dataset.

### Two issues with MAPS

NumPy's with the problematic samples described below are saved with 'WARN' prefix in the first IPython-template.

#### 1. Total silence at some time frames (currently not relevant).

Sometimes, depending on parameters, some cqt-columns are zeros (mainly for __*"ISOL"*__ samples, where loudness = __*"P"*__ (piano) & sustain pedal = __*"S0"*__ (not pressed)).  Don't know what the reason is, the corresponding WAV-samples sound perfectly fine.  When such columns fall into the time-frames we are interested in (regions of the note onsets), those samples are excluded, just to be safe (fortunately, there are usually just a few of such samples), and the message 'EXCLUDED' is printed in the first IPython-template.

Currently, the parameters of CQT-transform are such, that all cqts-columns are non-zero, and no samples are excluded.

#### 2. Two chords instead of one.

There is an issue with datasets __*"ENSTDkAm"*__ & __*"ENSTDkCl"*__, subtypes __*"RAND"*__ & __*"UCHO"*__.  They are assumed to have only one chord per one WAV-file.  But sometimes the chord is split into two onset times in corresponding MIDI and TXT-files, and those two onset times fall into two consecutive time-frames of cqt-transform.  BTW, __*"ENSTDkAm"*__ & __*"ENSTDkCl"*__ are "real piano" types, so, maybe it is the reason.

Not sure which was the correct way to handle this kind of samples.  Almost half of them are like this, so I did not want just to exclude them.  I ended up joining notes from two time-frames into one chord, and taking the cqt-data for the second onset time.  I printed 'WARNING' messages for such samples in the first IPython-template.


## International Audio Laboratories Erlangen

https://www.audiolabs-erlangen.de/resources/MIR/SMD/midi

This dataset is also quite big, so I included it too.

## Laboratory for the Recognition and Organization of Speech and Audio (LabROSA) Columbia

https://labrosa.ee.columbia.edu/projects/piano/

## Music and Audio Computing Lab, Vienna 4x22 corpus (Aligned)

http://mac.kaist.ac.kr/~ilcobo2/alignWithAMT/

These last two datasets, __*"LabROSA"*__ and __*"Vienna"*__, are much smaller, but I used them anyway.


# 1 Datasets Preparation

Constant-Q transform with the following parameters to play around:

1. Number of frequency bins per note (now it is 4).

	Currently, number of cqt bins is power of 2 for notes to be divisible by 2, so that we will be able to reshape the input in the way to capture higher harmonics more precisely.  We can always capture first three harmonics, but 4th and 5th are located 1.5 and 0.75 notes apart respectively.  However, not sure if it is really necessary, and maybe, 3 cqt bins would also be sufficient.

2. Number of time-frames before and after of note onset time to provide some context (now it is 3 before and 3 after).

	Probably, 2 before and 2 after would also be sufficient.

3. Harmonic-percussive separation margin and kernel size (currently librosa default).

	Don't know which would be optimal.
	__*Note:*__ cqts are passed to librosa.hpss, but it supposes to receive stft, so maybe, kernel height for percussive filter should be reduced proportionally (not done at the moment).

I also did non-linear logarithmic scaling of the cqt amplitude (librosa.amplitude_to_db).  Not sure if it has any influence on final accuracy though.

Also, normalization is important, without it the final program outputs would significantly depend on the volume of input.  But we would have to save normalization factors for using them on testing dataset.  So, instead of normalizing at pre-processing stage, the input is passed through the __*"BatchNormalization"*__ layer in CNN, it is more convenient and less error-prone.


# 2 Training and Validation

## 2.1 CNN Models

### Why Use Convolutions?

You may ask, can't we just suffice with fully-connected layers?  Yes, by the final layer of my model, every output neuron is influenced by every input bin.  Music transcription is simpler than images recognition, and it makes sense.  We do not need our Neural Network to learn small features, then bigger features and so on.  We already know pretty well which features and where we need to extract, but...

But we need to emphasize connections between a note and its harmonics.  Having just fully-connected layers would be overkill and would result in overfitting.  How can we tell fully-connected layer to keep only connections between harmonics and skip all other nodes?  Don't know, but it can easily be done with convolutions.  Actually, it can be done in a couple of ways.  So, there are two options for CNN-model described below.

### 2.1.1 CNN Model, Option 1.

To connect harmonics with each other, I used basically the following hack.  I reshaped input in such way that octaves lie beneath each other, and then sent it through the convolutional layer with filters aligned along harmonics.

Unfortunately, it is not enough.  Yes, 1st harmonic lies 12 notes apart from the fundamental frequency, but 2nd harmonic - 6 notes apart from the 1st, 3rd harmonic - 3 notes apart, etc.  So, I reshaped input several times, and finally had several fully-connected layers anyway (fully-connected layers speed up learning).

With this hack, small filter sizes are sufficient (I used 1x1 and 3x3).  Actually, I used the forward skip connections (__*"ResNet"*__) similar to the left variant (called __*"bottle neck"*__) from the figure below:

![](https://cdn-images-1.medium.com/max/1600/1*7JzJ1RGh1Y4VoG1M4dseTw.png 'ResNets')
With the variant from the right from the figure above, each epoch took too much time, and I did not have patience to wait.

### 2.1.2 CNN Model, Option 2.

"Option 1" is not bad, it just contains Keras __*"Permute"*__ layers which are currently not supported by __*"Frugally Deep"*__ Keras to C++ convertor (https://github.com/Dobiasd/frugally-deep).  Unfortunately, to correctly reshape the tensor, we need to have its axes in a specific order.  Then, to pass the tensor through the 2D-convolutional layer, we again need to have its "feature" axis (harmonics in our case) to be the last.  So, most of my reshapings were like __*"Permute --> Reshape --> Permute"*__.  To call that model predictions from C++, one would have to split the model in several parts and reshape tensors in between in C++ manually.

So, here is "Option 2", which can be straightforwardly converted to C++. The idea is entirely based on the algorithm explanation of the following commercial software:

https://www.lunaverus.com/cnn

1. *"... there's no need for fully connected layers and the CNN can be made entirely of convolutional and max pooling layers."*
2. *"... many of the convolutions long and skinny."*
3. *"... Most of the network consisted of pairs of layers: an Mx1 followed by a 1xN.  These long convolutions helped efficiently connect distant regions of the spectrum."*

So, it is just plain __*"ResNet"*__ layers, but with huge filter sizes which would cover all harmonics of interest. It would probably be better to use smaller filters and more layers (current trend for __*"ResNet"*__ models and image recognition).  Not sure if it would be applicable for audio transcription, but in this case model trains much slower anyway.  So, again, I did not have patience to wait and used short and wide __*"ResNet"*__.

Surprisingly, there is no sign of overfitting in this model, and dropout is not required at all :smile:  Also, since it is __*"ResNet"*__, and it should not become worse with the more number of layers, I repeated each residual block N number of times.  I tried N = 1, 2, 5, 10, 11, and sticked to 5 as a compromise between accuracy and speed.  However, I could have missed optimal number in between (__*TODO*__: try N = 3, 4 and 6, 7, 8, 9).  N = 1 and 2 is obviously not enough, and on the other hand, if N is too large, there is not much of improvement, plus, validation accuracy fluctuates more (maybe due to reduced batch size).


## 2.2 Validation

Standard accuracy is not representative, it would be almost 50% immediately at the beginning, and would obviously rise up to 99% very quickly.  F1-score is better, but even better is the percentage of samples (time-frames) where every single one of the 88 outputs (rounded to 0 or 1) is correct, I called it __*"Frame Accuracy"*__.  In other words, it is the percentage of chords which were predicted absolutely correct.

__*"Frame Accuracy"*__ of samples in the evaluation split (25% of data not used for training) came out to around 54%.  Macro-averaged F1-score =~ 73%. Micro-averaged F1-score would be slightly higher.


# 3 Piano Audio to Midi

No instrument information is extracted, and all transcribed notes get combined into one part.

There could be lots of noticeable false-positive notes in high octaves where they absolutely surely should not be.  If it is too annoying, then I cannot think of any better solution than to use much higher threshold for higher octaves :disappointed: