from tensorflow.keras.models import load_model
import tensorflow as tf
import sounddevice as sd
import numpy as np
import librosa

model = tf.keras.models.load_model('pruned_wake_word_model.h5')

SAMPLE_RATE = 16000            # Parameters
DURATION = 2  

def record_audio(duration):
    """Record audio for the given duration."""
    audio = sd.rec(int(SAMPLE_RATE * duration), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()                  # Wait until recording is finished
    return audio.flatten()

def preprocess_audio(audio):
    """Preprocess audio to extract MFCC features."""
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=13)
    return mfcc.T

def noise_reduction(audio):
    stft = librosa.stft(audio)
    magnitude, phase = np.abs(stft), np.angle(stft)
    noise_profile = np.mean(magnitude, axis=1, keepdims=True)
    threshold = 1.5 * noise_profile
    magnitude[magnitude < threshold] = 0
    stft_filtered = magnitude * np.exp(1j * phase)
    audio_filtered = librosa.istft(stft_filtered)
    return audio_filtered

print("Listening for the wake word... ")
try:
    while True:
        # Record audio
        audio = record_audio(DURATION)
        
        # Apply noise reduction
        audio = noise_reduction(audio)
        
        # Preprocess the audio
        mfcc_features = preprocess_audio(audio)
        
        # Pad or truncate the MFCC features to the required input shape
        mfcc_features = np.pad(mfcc_features, ((0, max(0, 32 - mfcc_features.shape[0])), (0, 0)), 'constant')
        mfcc_features = mfcc_features[:32]  # Ensure it's 32 time steps
        mfcc_features = mfcc_features.reshape(1, 32, 13, 1)  # Reshape for the model input

        # Make predictions
        prediction = model.predict(mfcc_features)
        if prediction[0][0] > 0.7:  # If detected wake word
            print("Detected")
except KeyboardInterrupt:
    print("Stopped listening.")
