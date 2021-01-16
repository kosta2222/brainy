import librosa
import os
from sklearn.model_selection import train_test_split
#from keras.utils import to_categorical
import numpy as np
from tqdm import tqdm
import pickle

DATA_PATH = "./data/"
NPY_PATH = "./"


# Input: Folder Path
# Output: Tuple (Label, Indices of the labels, one-hot encoded labels)
def get_labels(path=DATA_PATH, npy_path=NPY_PATH):
    labels = os.listdir(path)
    np.save(npy_path + 'labels.npy', labels)
    label_indices = np.arange(0, len(labels))
    return labels, label_indices


# Handy function to convert wav2mfcc
def wav2mfcc(file_path, max_len=11):
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    wave = wave[::3]
    wave = np.asfortranarray(wave)
    mfcc = librosa.feature.mfcc(wave, sr=16000)

    # If maximum length exceeds mfcc lengths then pad the remaining ones
    if (max_len > mfcc.shape[1]):
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')

    # Else cutoff the remaining parts
    else:
        mfcc = mfcc[:, :max_len]
    
    return mfcc


def save_data_to_array(path=DATA_PATH, max_len=11):
    labels, _, _ = get_labels(path)

    for label in labels:
        # Init mfcc vectors
        mfcc_vectors = []

        wavfiles = [path + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]
        for wavfile in tqdm(wavfiles, "Saving vectors of label - '{}'".format(label)):
            mfcc = wav2mfcc(wavfile, max_len=max_len)
            mfcc_vectors.append(mfcc)
        np.save(label + '.npy', mfcc_vectors)

def wav2mfcc_array(array, max_len=11):
    # wave, sr = librosa.load(file_path, mono=True, sr=None)
    wave = array[::3]
    wave = np.asfortranarray(wave)
    mfcc = librosa.feature.mfcc(wave, sr=16000)

    # If maximum length exceeds mfcc lengths then pad the remaining ones
    if (max_len > mfcc.shape[1]):
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')

    # Else cutoff the remaining parts
    else:
        mfcc = mfcc[:, :max_len]
    
    return mfcc    

def get_train_test(split_ratio=0.6, random_state=42):
    # Get available labels
    #labels, indices, _ = get_labels(DATA_PATH)
    labels=np.load(NPY_PATH +'labels.npy')
    indices = np.arange(0, len(labels))

    # Getting first arrays
    X = np.load(labels[0] + '.npy')
    y = []
    y_hot_0 = np.zeros(len(labels))
    y_hot_0[indices[0]] = 1
    
    X_height = X.shape[0]
    for i in range(X_height):
        y.append(list(y_hot_0))
    
    

    # Append all of the dataset into one single array, same goes for y
    for i, label in enumerate(labels[1:]):
        x = np.load(label + '.npy')
        X = np.vstack((X, x))

        X_height = x.shape[0]
        y_hot = np.zeros(len(labels))
        y_hot[indices[i + 1]] = 1    
        for i in range(X_height):    
           y.append(list(y_hot))

    y = np.array(y)    
    print('y', y)
    print('X shape', X.shape)

    # assert X.shape[0] == len(y)

    return train_test_split(X, y, test_size= (1 - split_ratio), random_state=random_state, shuffle=True)
    # return X, y



def prepare_dataset(path=DATA_PATH):
    labels, _, _ = get_labels(path)
    data = {}
    for label in labels:
        data[label] = {}
        data[label]['path'] = [path  + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]

        vectors = []

        for wavfile in data[label]['path']:
            wave, sr = librosa.load(wavfile, mono=True, sr=None)
            # Downsampling
            wave = wave[::3]
            mfcc = librosa.feature.mfcc(wave, sr=16000)
            vectors.append(mfcc)

        data[label]['mfcc'] = vectors

    return data


def load_dataset(path=DATA_PATH):
    data = prepare_dataset(path)

    dataset = []

    for key in data:
        for mfcc in data[key]['mfcc']:
            dataset.append((key, mfcc))

    return dataset[:100]


# print(prepare_dataset(DATA_PATH))

