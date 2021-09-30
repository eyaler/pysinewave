import sys

import numpy as np
import sounddevice as sd

from pysinewave import utilities
from pysinewave import sinewave_generator


class SineWave:
    '''Generates and plays a continuous sinewave, with smooth transitions in frequency (pitch)
        and amplitude (volume).'''

    def __init__(self, pitch=0, pitch_per_second=12, decibels=0, decibels_per_second=1, channels=1, channel_side="lr",
                 samplerate=utilities.DEFAULT_SAMPLE_RATE, clip_off=False, dither_off=False, waveform=np.sin,
                 phase_cutoff=2000000000, db_cutoff=-100):

        self.sinewave_generator = sinewave_generator.SineWaveGenerator(pitch=pitch, pitch_per_second=pitch_per_second,
                                                                       decibels=decibels,
                                                                       decibels_per_second=decibels_per_second,
                                                                       samplerate=samplerate, waveform=waveform,
                                                                       phase_cutoff=phase_cutoff, db_cutoff=db_cutoff)

        # Create the output stream
        self.output_stream = sd.OutputStream(channels=channels, callback=lambda *args: self._callback(*args),
                                             samplerate=samplerate, clip_off=clip_off, dither_off=dither_off)

        self.channels = channels

        if channel_side == 'r':
            self.channel_side = 0
        elif channel_side == 'l':
            self.channel_side = 1
        else:
            self.channel_side = -1

        self.data = None

    def _callback(self, outdata, frames, time, status):
        '''Callback function for the output stream.'''
        # Print any error messages we receive
        if status:
            print(status, file=sys.stderr)

        # Get and use the sinewave's next batch of data
        self.data = self.sinewave_generator.next_data(frames)
        outdata[:] = self.data.reshape(-1, 1)

        # Output on the given channel
        if self.channel_side != -1 and self.channels == 2:
            outdata[:, self.channel_side] = 0.0

    def play(self):
        '''Plays the sinewave (in a separate thread). Changes in frequency or amplitude will transition smoothly.'''
        self.output_stream.start()

    def stop(self):
        '''If the sinewave is playing, stops the sinewave.'''
        self.output_stream.stop()

    def set_frequency(self, frequency):
        '''Sets the goal frequency of the sinewave, which will be smoothly transitioned to.'''
        self.sinewave_generator.set_frequency(frequency)

    def set_pitch(self, pitch):
        '''Sets the goal pitch of the sinewave (relative to middle C),
        which will be smoothly transitioned to.'''
        self.sinewave_generator.set_pitch(pitch)

    def set_volume(self, volume):
        '''Sets the goal volume (in decibels, relative to medium volume) of the sinewave'''
        self.sinewave_generator.set_decibels(volume)

    def set_waveform(self, waveform):
        '''Sets the goal volume (in decibels, relative to medium volume) of the sinewave'''
        self.sinewave_generator.set_waveform(waveform)
