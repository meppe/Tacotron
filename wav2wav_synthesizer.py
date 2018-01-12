import io
import numpy as np
import tensorflow as tf
from hparams import hparams
from librosa import effects
from models import create_model
from text import text_to_sequence
from util import audio
from hparams import hparams, hparams_debug_string
import argparse
import os
import re


class Wav2WavSynthesizer:

  def load(self, checkpoint_path, model_name='wav2wav_tacotron'):
    print('Constructing model: %s' % model_name)
    inputs = tf.placeholder(tf.float32, [1, hparams.num_src_freq, None], 'inputs')
    input_lengths = tf.placeholder(tf.int32, [1], 'input_lengths')
    with tf.variable_scope('model') as scope:
      self.model = create_model(model_name, hparams)
      self.model.initialize(inputs, input_lengths)
      self.wav_output = audio.inv_spectrogram_tensorflow(self.model.linear_outputs[0])

    print('Loading checkpoint: %s' % checkpoint_path)
    self.session = tf.Session()
    self.session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(self.session, checkpoint_path)

  def synthesize(self, in_file):
    src_spectrogram = audio.spectrogram(in_file,
                                        num_src_freq=hparams.num_src_freq,
                                        frame_length_ms=hparams.src_frame_length_ms).astype(np.float32)
    feed_dict = {
      self.model.inputs: [np.asarray(src_spectrogram, dtype=np.float32)],
      self.model.input_lengths: np.asarray([len(src_spectrogram)], dtype=np.int32)
    }
    wav = self.session.run(self.wav_output, feed_dict=feed_dict)
    wav = audio.inv_preemphasis(wav)
    wav = wav[:audio.find_endpoint(wav)]
    out = io.BytesIO()
    audio.save_wav(wav, out)
    return out.getvalue()

  def get_output_base_path(self, checkpoint_path):
    base_dir = os.path.dirname(checkpoint_path)
    m = re.compile(r'.*?\.ckpt\-([0-9]+)').match(checkpoint_path)
    name = 'eval-%d' % int(m.group(1)) if m else 'eval'
    return os.path.join(base_dir, name)

if __name__ == '__main__':
  w2w = Wav2WavSynthesizer()
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', default=1000, help='Path to model checkpoint')
  parser.add_argument('-i', required=True, help='Path to input wave file', dest="in_file")
  parser.add_argument('--hparams', default='',
    help='Hyperparameter overrides as a comma-separated list of name=value pairs')
  args = parser.parse_args()
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  hparams.parse(args.hparams)
  w2w.load(args.checkpoint)
  base_path = w2w.get_output_base_path(args.checkpoint)
  if not os.path.exists(args.in_file):
    print("file {} does not exist. Existing.".format(args.in_file))
  else:
    out_path = '%s-out.wav' % (base_path)
    print('Synthesizing: %s' % out_path)
    with open(out_path, 'wb') as f:
      f.write(w2w.synthesize(args.in_file))

    print("done")

