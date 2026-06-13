#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Detects onsets, beats and tempo in WAV files.

For usage information, call with --help.

Author: Jan Schlüter
"""

import sys
from pathlib import Path
from argparse import ArgumentParser
import json

import numpy as np
from librosa import iirt
from librosa.filters import semitone_filterbank
from scipy.io import wavfile
import librosa
from scipy.ndimage import maximum_filter, maximum_filter1d, median_filter

try:
    import tqdm
except ImportError:
    tqdm = None


def opts_parser():
    usage =\
"""Detects onsets, beats and tempo in WAV files.
"""
    parser = ArgumentParser(description=usage)
    parser.add_argument('indir',
            type=str,
            help='Directory of WAV files to process.')
    parser.add_argument('outfile',
            type=str,
            help='Output JSON file to write.')
    parser.add_argument('--plot',
            action='store_true',
            help='If given, plot something for every file processed.')
    return parser


def detect_everything(filename, options):
    """
    Computes some shared features and calls the onset, tempo and beat detectors.
    """
    print(filename)
    # read wave file (this is faster than librosa.load)
    sample_rate, signal = wavfile.read(filename)

    # convert from integer to float
    if signal.dtype.kind == 'i':
        signal = signal / np.iinfo(signal.dtype).max

    # convert from stereo to mono (just in case)
    if signal.ndim == 2:
        signal = signal.mean(axis=-1)

    # compute spectrogram with given number of frames per second
    fps = 70
    hop_length = sample_rate // fps
    spect = librosa.stft(
            signal, n_fft=2048, hop_length=hop_length, window='hann')

    # only keep the magnitude
    magspect = np.abs(spect)

    # compute a mel spectrogram
    melspect = librosa.feature.melspectrogram(
            S=magspect, sr=sample_rate, n_mels=80, fmin=27.5, fmax=8000)

    # compress magnitudes logarithmically
    melspect = np.log1p(100 * melspect)

    # compute onset detection function
    odf, odf_rate = onset_detection_function(
            sample_rate, signal, fps, spect, magspect, melspect, options)

    # detect onsets from the onset detection function
    onsets = detect_onsets(odf_rate, odf, options)

    # detect tempo from everything we have
    tempo = detect_tempo(
            sample_rate, signal, fps, spect, magspect, melspect,
            odf_rate, odf, onsets, options)

    # detect beats from everything we have (including the tempo)
    beats = detect_beats(
            sample_rate, signal, fps, spect, magspect, melspect,
            odf_rate, odf, onsets, tempo, options)

    # plot some things for easier debugging, if asked for it
    if options.plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, sharex=True)
        plt.subplots_adjust(hspace=0.3)
        plt.suptitle(filename)
        axes[0].set_title('melspect')
        axes[0].imshow(melspect, origin='lower', aspect='auto',
                       extent=(0, melspect.shape[1] / fps,
                               -0.5, melspect.shape[0] - 0.5))
        axes[1].set_title('onsets')
        axes[1].plot(np.arange(len(odf)) / odf_rate, odf)
        for position in onsets:
            axes[1].axvline(position, color='tab:orange')
        axes[2].set_title('beats (tempo: %r)' % list(np.round(tempo, 2)))
        axes[2].plot(np.arange(len(odf)) / odf_rate, odf)
        for position in beats:
            axes[2].axvline(position, color='tab:red')
        plt.show()

    return {'onsets': list(np.round(onsets, 3)),
            'beats': list(np.round(beats, 3)),
            'tempo': list(np.round(tempo, 2))}


def onset_detection_function(sample_rate, signal, fps, spect, magspect,
                             melspect, options):
    """
    Compute an onset detection function. Ideally, this would have peaks
    where the onsets are. Returns the function values and its sample/frame
    rate in values per second as a tuple: (values, values_per_second)
    """
    return log_filt_spec_flux_detection_function(sample_rate, signal, fps, spect, magspect,
                              melspect, options)

def high_frequency_component_detection_function(sample_rate, signal, fps, spect, magspect, melspect, options):
    energy_squared = melspect ** 2 # |X_k[t]| ^ 2
    n_bins = energy_squared.shape[0]
    weights = np.arange(n_bins) ** 2 # W_k = k; higher bin, higher weight
    weighted_energy_squared = energy_squared * weights[:, np.newaxis] # colum-wise multiplication on each bin
    values = np.sum(weighted_energy_squared, axis=0) # sum over all bins
    values = values / n_bins

    return values, fps

def phase_difference_detection_function(sample_rate, signal, fps, spect, magspect, melspect, options):
    # get phase and convert to cycles
    phase = np.angle(spect)/(2*np.pi)

    # get first order of phase
    phase_first_order=phase[:,1:]-phase[:,:-1]
    # wrap values
    phase_first_order = ((phase_first_order + 0.5) % 1) - 0.5
    # get second order
    phase_second_order=phase_first_order[:,1:]-phase_first_order[:,:-1]
    phase_second_order = ((phase_second_order + 0.5) % 1) - 0.5
    # get phase second order per frame over whole frequency spectrum
    pso_per_frame=np.sum(magspect[:,2:] * np.abs(phase_second_order), 0)
    values = pso_per_frame

    # normalize and return
    if np.max(values)>0:
        values=values/np.max(values)
    values_per_second = fps
    return values, values_per_second

def log_filt_spec_flux_detection_function(sample_rate, signal, fps, spect, magspect, melspect, options):
    # Preprocessing
    hop_length = sample_rate // fps
    # get semitone spectogram, returns mean-squared power
    sem_spect = iirt(signal, sr=sample_rate, hop_length=hop_length)
    # convert to magnitude
    sem_spect = np.sqrt(sem_spect)
    # convert to logarithmic magnitude
    lambda_par = 17
    sem_spect = np.log1p(lambda_par * sem_spect)

    # Detection Function
    differences = np.diff(sem_spect)
    # apply half-wave rectifier
    differences = np.maximum(differences, 0)
    values = np.sum(differences, axis=0)
    return values, fps

def detect_onsets(odf_rate, odf, options):
    """
    Detect onsets in the onset detection function.
    Returns the positions in seconds.
    """
    return detect_onsets_lfsf(odf_rate, odf, options)

def detect_onsets_phase_difference(odf_rate, odf, options):
    """
    Detect onsets in the onset detection function.
    Returns the positions in seconds.
    """
    # smooth odf by convolving over function
    window_size = np.ones(10) / 10
    peaks_smoothed = np.convolve(odf, window_size, mode='same')
    # detect peaks in odf using standard deviation as delta
    delta = 0.19 * np.std(odf)
    peaks = np.where((odf[1:-1] > odf[:-2])
                     & (odf[1:-1] > odf[2:])
                     & (odf[1:-1] > peaks_smoothed[1:-1] + delta))
    # correct offset (for phase difference) and transform to seconds
    onsets = (peaks[0] + 1.0) / odf_rate
    if len(onsets) == 0:
        return []
    # 50 ms pause between peaks
    onsets_minimum_gap = [onsets[0]]
    for x in onsets[1:]:
        if x - onsets_minimum_gap[-1] >= 0.05:
            onsets_minimum_gap.append(x)
    return onsets_minimum_gap

def detect_onsets_lfsf(odf_rate, odf, options):
    delta = 0.46
    window_size = 11

    # get maxima and mean in window
    local_maxima = maximum_filter(odf, window_size)
    local_mean = median_filter(odf, window_size)

    # get possible peak positions
    is_peak_maxima = odf == local_maxima
    is_peak_mean = odf >= local_mean + delta
    is_peak_mask = is_peak_maxima & is_peak_mean
    peak_frames = np.where(is_peak_mask)[0]

    # convert position into time domain
    onsets = np.array(peak_frames / odf_rate)
    if len(onsets) == 0:
        return []

    # filter out peaks that are too close
    onsets_minimum_gap = [onsets[0]]
    for x in onsets[1:]:
        if x - onsets_minimum_gap[-1] >= 0.05:
            onsets_minimum_gap.append(x)

    return onsets_minimum_gap

def detect_tempo(sample_rate, signal, fps, spect, magspect, melspect,
                 odf_rate, odf, onsets, options):
    """
    Detect tempo using any of the input representations.
    Returns one tempo or two tempo estimations.
    """
    # we only have a dumb dummy implementation here.
    # it uses the time difference between the first two onsets to
    # define the tempo, and returns half of that as a second guess.
    # this is not a useful solution at all, just a placeholder.
    tempo = 60 / (onsets[1] - onsets[0])
    return [tempo / 2, tempo]


def detect_beats(sample_rate, signal, fps, spect, magspect, melspect,
                 odf_rate, odf, onsets, tempo, options):
    """
    Detect beats using any of the input representations.
    Returns the positions of all beats in seconds.
    """
    # we only have a dumb dummy implementation here.
    # it returns every 10th onset as a beat.
    # this is not a useful solution at all, just a placeholder.
    return onsets[::10]


def main():
    # parse command line
    parser = opts_parser()
    options = parser.parse_args()

    # iterate over input directory
    indir = Path(options.indir)
    infiles = list(indir.glob('*.wav'))
    if tqdm is not None:
        infiles = tqdm.tqdm(infiles, desc='File')
    results = {}
    for filename in infiles:
        results[filename.stem] = detect_everything(filename, options)

    # write output file
    with open(options.outfile, 'w') as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()

