# Wake-word-detection
Lightweight model for wake word detection

This pruned lightweight wake word detection model is designed for efficient performance on resource-constrained edge devices. Used LSTM (Long Short-Term Memory) layers to capture temporal dependencies in audio data, which is crucial for accurately detecting wake words in continuous speech. The model uses Mel-frequency cepstral coefficients (MFCCs) to extract meaningful features from the audio signals, enhancing its ability to distinguish between wake words and background noise.

Pruning has been applied to reduce the size and complexity of the LSTM layers by removing redundant parameters, leading to faster inference times and lower memory usage. Noise augmentation during training improves robustness, allowing the model to perform well even in varied environments. Overall, this lightweight LSTM-based model balances accuracy with efficiency, making it ideal for real-time wake word detection on low-power devices such as smartphones and embedded systems.
