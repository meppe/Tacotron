from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import os
from util import audio
from hparams import hparams


def build_from_path(in_dir, out_dir, num_workers=1, tqdm=lambda x: x):
  '''Preprocesses the LJ Speech dataset from a given input path into a given output directory.

    Args:
      in_dir: The directory where you have downloaded the LJ Speech dataset
      out_dir: The directory to write the output into
      num_workers: Optional number of worker processes to parallelize across
      tqdm: You can optionally pass tqdm to get a nice progress bar

    Returns:
      A list of tuples describing the training examples. This should be written to train.txt
  '''

  # We use ProcessPoolExecutor to parallize across processes. This is just an optimization and you
  # can omit it and just call _process_utterance on each input if you want.
  executor = ProcessPoolExecutor(max_workers=num_workers)
  futures = []
  index = 1
  files = [f for f in os.listdir(in_dir+"/src/") if f.find(".wav") >= 0]
  for fname in files:
    src_path = in_dir + "/src/" + fname
    tgt_path = in_dir + "/tgt/" + fname
    futures.append(executor.submit(partial(_process_utterance, out_dir, index, src_path, tgt_path)))
    index += 1
  return [future.result() for future in tqdm(futures)]


def _process_utterance(out_dir, index, src_path, tgt_path):
  '''Preprocesses a single utterance audio/text pair.

  This writes the mel and linear scale spectrograms to disk and returns a tuple to write
  to the train.txt file.

  Args:
    out_dir: The directory to write the spectrograms into
    index: The numeric index to use in the spectrogram filenames.
    src_path: Path to the source audio file
    tgt_path: Path to the target audio file

  Returns:
    A (tgt_spectrogram_filename, tgt_mel_filename, n_frames, src_spectogram_filename) tuple to write to train.txt
  '''

  # Load the audio to a numpy array:
  src_wav = audio.load_wav(src_path)
  tgt_wav = audio.load_wav(tgt_path)

  # Compute the linear-scale spectrogram from the wav:
  src_spectrogram = audio.spectrogram(src_wav,
                                      num_src_freq=hparams.num_src_freq,
                                      frame_length_ms=hparams.src_frame_length_ms).astype(np.float32)
  src_n_frames = src_spectrogram.shape[1]
  tgt_spectrogram = audio.spectrogram(tgt_wav).astype(np.float32)
  tgt_n_frames = tgt_spectrogram.shape[1]

  # Compute a mel-scale spectrogram from the wav:
  src_mel_spectrogram = audio.melspectrogram(src_wav).astype(np.float32)
  tgt_mel_spectrogram = audio.melspectrogram(tgt_wav).astype(np.float32)

  # Write the spectrograms to disk:
  src_spectrogram_filename = 'wav2wav_src-spec-%05d.npy' % index
  src_mel_filename = 'wav2wav_src-mel-%05d.npy' % index
  np.save(os.path.join(out_dir, src_spectrogram_filename), src_spectrogram.T, allow_pickle=False)
  np.save(os.path.join(out_dir, src_mel_filename), src_mel_spectrogram.T, allow_pickle=False)

  tgt_spectrogram_filename = 'wav2wav_tgt-spec-%05d.npy' % index
  tgt_mel_filename = 'wav2wav_tgt-mel-%05d.npy' % index
  np.save(os.path.join(out_dir, tgt_spectrogram_filename), tgt_spectrogram.T, allow_pickle=False)
  np.save(os.path.join(out_dir, tgt_mel_filename), tgt_mel_spectrogram.T, allow_pickle=False)

  # Return a tuple describing this training example:
  return (tgt_spectrogram_filename, tgt_mel_filename, tgt_n_frames, src_spectrogram_filename)
