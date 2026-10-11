# 🎵 AI Music Generation using LSTM

A deep learning project that learns patterns from classical piano MIDI files and generates new music using an LSTM (Long Short-Term Memory) neural network.

## 📌 Overview

The model is trained on note sequences extracted from classical piano compositions. Given 50 notes, it predicts the next note, and repeating this step produces a brand-new piece of music. The generated notes are converted back to a MIDI file and then to a WAV audio file.

## 🔄 Pipeline

1. **Data collection:** Classical piano MIDI files (19 composers, 295 files)
2. **Preprocessing:** Notes and chords extracted with `music21` (150 files used, 239,806 notes/chords, vocabulary of 529 unique notes/chords)
3. **Sequence creation:** Input = 50 notes, output = the next note (239,756 training sequences)
4. **Model:** Embedding → LSTM (256) → Dropout → LSTM (256) → Dropout → Dense (256) → Dropout → Softmax
5. **Training:** 20 epochs, batch size 256, Adam optimizer, on Google Colab (T4 GPU)
6. **Generation:** Temperature-based sampling (temperature = 0.8), 300 notes
7. **Output:** Notes converted to MIDI (`music21`) and audio (`pretty_midi` + FluidSynth)

## 🧠 Model Summary

| Layer | Output Shape | Params |
|---|---|---|
| Embedding | (None, 50, 100) | 52,900 |
| LSTM | (None, 50, 256) | 365,568 |
| LSTM | (None, 256) | 525,312 |
| Dense | (None, 256) | 65,792 |
| Dense (softmax) | (None, 529) | 135,953 |

Total parameters: **1,145,525**

## 📉 Training Result

Loss went down from **4.65 (epoch 1)** to **2.76 (epoch 20)**.

## 📁 Files in this repository

| File | Description |
|---|---|
| `Music_Generation_LSTM.ipynb` | Full notebook (preprocessing, training, generation) |
| `generated_music.mid` | AI-generated music (MIDI) |
| `generated_music.wav` | AI-generated music (audio) |
| `music_model.keras` | Trained LSTM model |

## 🛠️ Tech Stack

Python, TensorFlow/Keras, music21, pretty_midi, FluidSynth, NumPy, Google Colab

## ▶️ How to run

1. Open `Music_Generation_LSTM.ipynb` in Google Colab
2. Set Runtime to **T4 GPU**
3. Add the dataset (see below) to your Google Drive and run the cells in order

## 📚 Dataset

[Classical Music MIDI](https://www.kaggle.com/datasets/soumikrakshit/classical-music-midi) by Soumik Rakshit on Kaggle (files originally from piano-midi.de). The dataset is not included in this repository, please download it from Kaggle.

## 🚀 Future Improvements

- Train for more epochs and on more data
- Add jazz and other genres
- Try a Transformer or GAN based model
- Predict note duration and velocity, not only pitch

## 👩‍💻 Author

**Zainab Batool**
GitHub: [@zainabbatool-bit](https://github.com/zainabbatool-bit)
LinkedIn: [Zainab Batool](https://www.linkedin.com/in/zainab-batool-b4661734a)
