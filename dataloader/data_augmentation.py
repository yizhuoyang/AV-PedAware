import numpy as np
import matplotlib.pyplot as plt


def add_gaussian_noise(audiodata, target_snr_db):
    """
    Add Gaussian noise to a multi-channel audio signal at a specified SNR level.

    :param audiodata: Original multi-channel audio (shape: (num_mics, num_samples))
    :param target_snr_db: Target SNR level in dB (same for all mics)
    :return: Noisy audio with the specified SNR
    """
    num_mics, num_samples = audiodata.shape
    noisy_audio = np.zeros_like(audiodata)

    for i in range(num_mics):
        signal = audiodata[i]  # Extract one mic channel
        noise = np.random.normal(0, 1, num_samples)  # Generate Gaussian noise

        # Compute power of signal and noise
        signal_power = np.mean(signal ** 2)
        noise_power = np.mean(noise ** 2)

        # Compute noise scaling factor to match desired SNR
        snr_scaling_factor = np.sqrt(signal_power / (10 ** (target_snr_db / 10) * noise_power))

        # Scale and add noise
        noise = noise * snr_scaling_factor
        noisy_audio[i] = signal + noise  # Add noise to original signal

    return noisy_audio

def simulate_microphone_desync(audio_data, fs, max_time_offset_ms):
    """
    Simulates microphone array desynchronization by applying different time offsets.

    :param audio_data: Original multi-channel audio, shape (num_mics, num_samples)
    :param fs: Sampling rate (Hz)
    :param max_time_offset_ms: Maximum time offset (milliseconds) between microphones
    :return: Time-desynchronized audio data
    """
    num_mics, num_samples = audio_data.shape
    desynced_audio = np.zeros_like(audio_data)

    # Convert max offset to sample delay
    max_sample_offset = int((max_time_offset_ms / 1000) * fs)

    # Generate random time offsets for each microphone
    time_offsets = np.random.randint(0, max_sample_offset + 1, num_mics)  # Different offset for each mic
    # print(f"🛠 Time offsets (samples): {time_offsets}")

    for i in range(num_mics):
        delay = time_offsets[i]
        desynced_audio[i, delay:] = audio_data[i, :-delay] if delay > 0 else audio_data[i]  # Shift signal

    return desynced_audio, time_offsets

if __name__ == "__main__":
    ## The function for SNR simulation
    fs = 16000  # Sampling rate
    audiodata = np.load('/media/kemove/T9/sound_source_loc/simulation_data/train/coherent/NS_1/degree_0.0__times0.npy')  # Simulated audio data (4 mics, 1 sec)
    target_snr_db = -10  # Set SNR level
    # Generate noisy audio
    noisy_audio = add_gaussian_noise(audiodata, target_snr_db)
    # Select a single microphone channel for visualization
    mic_channel = 0
    original_signal = audiodata[mic_channel]
    noisy_signal = noisy_audio[mic_channel]

    # Plot the original and noisy waveforms
    plt.figure(figsize=(12, 5))
    plt.subplot(2, 1, 1)
    plt.plot(original_signal, label="Original Signal", color="blue")
    plt.title("Original Signal (Time Domain)")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(noisy_signal, label=f"Noisy Signal (SNR={target_snr_db} dB)", color="red")
    plt.title("Noisy Signal (Time Domain)")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.show()
#
# ## The function for delay simulation
#     # Example Usage
#     fs = 16000  # 16 kHz sampling rate
#     duration = 1  # 1 second
#     num_samples = fs * duration
#     num_mics = 4  # 4 microphone channels
#
#     # freq = 440  # 440 Hz (A4 note)
#     time = np.linspace(0, duration, num_samples, endpoint=False)
#     # audio_data = np.sin(2 * np.pi * freq * time)  # Single sine wave
#     # audio_data = np.tile(audio_data, (num_mics, 1))  # Replicate across 4 microphones
#     audio_data = audiodata
#     # Simulate time desynchronization (max 3 ms)
#     max_time_offset_ms = 3
#     desynced_audio, time_offsets = simulate_microphone_desync(audio_data, fs, max_time_offset_ms)
#
#     # Plot comparison of original vs. desynced signal for first two mics
#     plt.figure(figsize=(10, 5))
#     plt.plot(time[:500], audio_data[0, :500], label="Mic 1 (Reference)", color="blue")
#     plt.plot(time[:500], desynced_audio[3, :500], label=f"Mic 2 (Offset={time_offsets[1]} samples)", linestyle="dashed", color="red")
#     plt.xlabel("Time (s)")
#     plt.ylabel("Amplitude")
#     plt.title("Microphone Time Desynchronization Simulation")
#     plt.legend()
#     plt.show()

