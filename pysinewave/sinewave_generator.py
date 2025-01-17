import numpy as np

from pysinewave import utilities


class SineWaveGenerator:
    """Generates continuous sine wave data that smoothly transitions between pitches and volumes.
    (For simplicity, use SineWave instead. 
    SineWaveGenerator is included to allow for alternative uses of generated sinewave data.)"""

    def __init__(self, pitch, pitch_per_second=12, decibels=1, decibels_per_second=1,
                 samplerate=utilities.DEFAULT_SAMPLE_RATE, waveform=np.sin, phase_cutoff=2000000000, db_cutoff=-100):
        self.frequency = utilities.pitch_to_frequency(pitch)
        self.phase = 0
        self.amplitude = utilities.decibels_to_amplitude_ratio(decibels)

        self.pitch_per_second = pitch_per_second
        self.decibels_per_second = decibels_per_second
        self.goal_frequency = self.frequency
        self.goal_amplitude = self.amplitude
        self.samplerate = samplerate
        self.waveform = waveform
        self.phase_cutoff = phase_cutoff
        self.amplitude_cutoff = utilities.decibels_to_amplitude_ratio(db_cutoff)

    def new_frequency_array(self, time_array):
        """Calcululate the frequency values for the next chunk of data."""
        dir = utilities.direction(self.frequency, self.goal_frequency)
        new_frequency = self.frequency * utilities.interval_to_frequency_ratio(
            dir * self.pitch_per_second * time_array)
        return utilities.bounded_by_end(new_frequency, self.frequency, self.goal_frequency)

    def new_amplitude_array(self, time_array):
        """Calcululate the amplitude values for the next chunk of data."""
        dir = utilities.direction(self.amplitude, self.goal_amplitude)
        new_amplitude = self.amplitude * utilities.decibels_to_amplitude_ratio(
            dir * self.decibels_per_second * time_array)
        new_amplitude = utilities.bounded_by_end(new_amplitude, self.amplitude, self.goal_amplitude)
        new_amplitude[new_amplitude <= self.amplitude_cutoff] = 0
        return new_amplitude

    def new_phase_array(self, new_frequency_array, delta_time):
        """Calcululate the phase values for the next chunk of data, given frequency values"""
        return self.phase + np.cumsum(new_frequency_array * delta_time)

    def set_frequency(self, frequency):
        """Set the goal frequency that the sinewave will gradually shift towards."""
        self.goal_frequency = frequency

    def set_pitch(self, pitch):
        """Set the goal pitch that the sinewave will gradually shift towards."""
        self.goal_frequency = utilities.pitch_to_frequency(pitch)

    def set_amplitude(self, amplitude):
        """Set the amplitude that the sinewave will gradually shift towards."""
        self.goal_amplitude = amplitude

    def set_waveform(self, waveform):
        self.waveform = waveform

    def set_decibels(self, decibels):
        """Set the amplitude (in decibels) that the sinewave will gradually shift towards."""
        self.goal_amplitude = utilities.decibels_to_amplitude_ratio(decibels)

    def next_data(self, frames):
        """Get the next pressure array for the given number of frames"""

        # Convert frame information to time information
        time_array = utilities.frames_to_time_array(0, frames, self.samplerate)
        delta_time = time_array[1] - time_array[0]

        # Calculate the frequencies of this batch of data
        new_frequency_array = self.new_frequency_array(time_array)

        # Calculate the phases
        new_phase_array = self.new_phase_array(new_frequency_array, delta_time)

        # Calculate the amplitudes
        new_amplitude_array = self.new_amplitude_array(time_array)

        # Create the sinewave array
        sinewave_array = new_amplitude_array * self.waveform(2 * np.pi * new_phase_array)

        # Update frequency and amplitude
        self.frequency = new_frequency_array[-1]
        self.amplitude = np.maximum(new_amplitude_array[-1], self.amplitude_cutoff)

        # Update phase (getting rid of extra cycles, so we don't eventually have an overflow error)
        self.phase = new_phase_array[-1]
        if self.phase > self.phase_cutoff:  # changed from phase_cutoff=1 to this to allow dealing with additive waveforms without accumulating an audible error within a period of a few hours
            self.phase -= self.phase_cutoff

        # print('Frequency: {0} Phase: {1} Amplitude: {2}'.format(self.frequency, self.phase, self.amplitude), np.max(np.abs(sinewave_array)))

        return sinewave_array
