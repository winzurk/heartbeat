import librosa
import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from tqdm import tqdm

# Shuffles X and y in unison :)
def shuffle_in_unison(a, b):
    np.random.seed(69) 
    rng_state = np.random.get_state()
    np.random.seed(69)
    np.random.shuffle(a)
    np.random.seed(69)
    np.random.set_state(rng_state)
    np.random.seed(69)
    np.random.shuffle(b)

# Input: Folder Path
# Output: Tuple (Label, Indices of the labels, one-hot encoded labels)
def get_labels(path):
    labels = os.listdir(path)
    label_indices = np.arange(0, len(labels))
    return labels, label_indices, to_categorical(label_indices)

# Input: Wave File
# Output: Mel Spectrogram
def wav2spec(file_path, max_seconds=3):
    wave, sr = librosa.load(file_path, mono=True, sr=2000)
    wave_max_len = 2000 * max_seconds #3 seconds
    wave = wave[:wave_max_len]
    spec = librosa.core.stft(wave, n_fft=256, hop_length=128, win_length=128, window='hann', center=True, pad_mode='reflect')
    return spec

# Input: Wave File
# Output: 1D Vector
def wav2vec(file_path, max_seconds):
    wave, sr = librosa.load(file_path, mono=True, sr=1000)
    wave_max_len = 1000 * max_seconds
    wave = wave[:wave_max_len]
    return wave

# Input: Wave File
# Output: MFCC
def wav2mfcc(file_path, max_seconds=3, n_mfcc=20):
    wave, sr = librosa.load(file_path, mono=True, sr=2000)
    wave_max_len = 2000 * max_seconds
    wave = wave[:wave_max_len]
    mfcc = librosa.feature.mfcc(y=wave, sr=sr, n_mfcc=n_mfcc, n_fft=512, hop_length=128)
    return mfcc

# Input: Data Path
# Saves Vectors  
def save_data_to_vec_tensor(path, max_seconds, dest_dir_path):

    labels = os.listdir(path)

    for label in labels:
        # Empty tensor
        tensors = []

        wavfiles = [path + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]
        for wavfile in tqdm(wavfiles, "Saving vectors of label - '{}'".format(label)):
            vec = wav2vec(wavfile, max_seconds=max_seconds)
            tensors.append(vec)
               
        #return tensors
        #output = np.stack(tensors, axis=0)
        np.save(dest_dir_path + '/' + label + '.npy', tensors)

# Input: Data Path
# Saves Data tensor of MFCC features  
def save_data_to_mfcc_tensor(path, max_seconds, dest_dir_path):

    labels = os.listdir(path)

    for label in labels:
        # Empty tensor
        tensors = []

        wavfiles = [path + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]
        for wavfile in tqdm(wavfiles, "Saving vectors of label - '{}'".format(label)):
            mfcc = wav2mfcc(wavfile, max_seconds=max_seconds)
            tensors.append(mfcc)
               
        #return tensors
        #output = np.stack(tensors, axis=0)
        np.save(dest_dir_path + '/' + label + '.npy', tensors)


# Input: Data Path
# Saves Data tensor of MelSpec features  
def save_data_to_spec_tensor(path, max_seconds, dest_dir_path):

    labels = os.listdir(path)

    for label in labels:
        # Empty tensor
        tensor = []

        wavfiles = [path + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]
        for wavfile in tqdm(wavfiles, "Saving vectors of label - '{}'".format(label)):
            spec = wav2spec(wavfile, max_seconds=max_seconds)
            tensor.append(spec)
        np.save(dest_dir_path + '/' + label + '.npy', tensor)


def load_tensor_from_numpy(data_path, numpy_path):

    labels, indices, _ = get_labels(data_path)

    X = np.load(numpy_path + labels[0] + '.npy')
    y = np.zeros(X.shape[0])

    for i, label in enumerate(labels[1:]):
        x = np.load(numpy_path + label + '.npy')
        X = np.vstack((X, x))
        y = np.append(y, np.full(x.shape[0], fill_value=(i+1)))

    assert X.shape[0] == len(y)

    #shuffle_in_unison(X, y)

    return X, y


def get_train_test(split_ratio=0.6, random_state=42):
    # Get available labels
    labels, indices, _ = get_labels(DATA_PATH)

    # Getting first arrays
    X = np.load(labels[0] + '.npy')
    y = np.zeros(X.shape[0])

    # Append all of the dataset into one single array, same goes for y
    for i, label in enumerate(labels[1:]):
        x = np.load(label + '.npy')
        X = np.vstack((X, x))
        y = np.append(y, np.full(x.shape[0], fill_value= (i + 1)))

    assert X.shape[0] == len(y)

    return train_test_split(X, y, test_size= (1 - split_ratio), random_state=random_state, shuffle=True)



