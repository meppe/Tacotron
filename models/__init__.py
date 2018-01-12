from .tacotron import Tacotron
from .wav2wav_tacotron import Wav2WavTacotron


def create_model(name, hparams):
  if name == 'tacotron':
    return Tacotron(hparams)
  elif name == 'wav2wav_tacotron':
    return Wav2WavTacotron(hparams)
  else:
    raise Exception('Unknown model: ' + name)
