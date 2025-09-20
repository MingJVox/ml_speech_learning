# 🧠 ML Speech Learning | Speech + Machine Learning Study Record

**Language**: English | [中文](README.md)

This repository is used to systematically record the entire process of learning machine learning in the speech domain from scratch, including basic courses, practical projects, container training, daily summary check-ins, and other content. The goal is to complete the transition from beginner to practitioner through 2-3 weeks of intensive learning, and accumulate project experience and code results for future research or internships.
The initial learning plan and resources came from discussions with ChatGPT, so I don't know what the results will be after 15 days either. This can also serve as an experiment in using ChatGPT for learning - let's wait and see. If anyone wants to learn together following this plan, please let me know so I know I'm not alone.. ❤️😭💪🏻

---

# Machine Learning & Speech Project Learning

## 🎯 Learning Goals
- Master PyTorch fundamentals
- Independently train CNN, speech classification, ASR, Voice Cloning, Speaker Verification
- Understand gradients, loss, hyperparameter tuning, and visualizing training processes
- Learn to train in Mac and GPU container environments

## 📅 Learning Plan
- 15-day rapid hands-on plan
- 8 hours daily, including video learning + coding practice + hyperparameter logging
- Project list:
  1. PyTorch Basics
  2. CNN Image Classification (MNIST / CIFAR10)
  3. Speech Classification (Speech Commands)
  4. ASR Wav2Vec2
  5. Voice Cloning
  6. Speaker Verification + Enhancement

## 📁 Repository Structure
- One folder per day, containing notebooks / hyperparameter tables / study notes / visualization plots
- Day 15 summary of experiences and visualization results
```
ml_speech_learning/
├── README.md              # Project homepage description
├── Day1/                  # Daily learning task records
│   ├── notebook.ipynb     # Learning code + experiment records
│   └── notes.md           # Summary / reflection / links / questions
├── Day2/ ~ Day15/         # Other daily tasks (to be added)
├── utils/                 # Common utility functions, e.g., feature extraction
├── containers/            # Dockerfile / Apptainer configuration files
├── .gitignore             # Ignore unnecessary files (like .DS_Store)
└── LICENSE                # Project license (MIT)
```

## 🧑‍💻 Daily Learning Method

Each `DayX/` folder should contain:

- `notebook.ipynb`: Code corresponding to the learning topic (see Day1 example)
- Day1 example directory template:
```
Day1_PyTorch_Basics/
├── 01_notebook.ipynb # Jupyter Notebook file
├── 02_loss_curve.png # Training curve screenshot
├── 03_params_table.csv # Hyperparameter tuning record table
└── 04_notes.md # Study notes (concepts + problems encountered + solutions)
```

- `notes.md`: Daily learning summary, problems encountered, links, reflections, etc.
- Day1 example directory template:
- Notebook file naming convention:
```
DayX_ProjectName_Description.ipynb
DayX_ProjectName_params.csv
DayX_ProjectName_loss.png
DayX_ProjectName_notes.md
```

## ⚡ Usage Instructions
- Can directly open and run Jupyter Notebooks
- Hyperparameter results and charts can be viewed in `*_curve.png` or `params_table.csv`
- After completing each day, commit updates:

```bash
git add .
git commit -m "Add DayX study files"
git push origin main
```

---

## ✏️ Daily Plan (8 hours per day)

Goal: Rapidly master PyTorch, hyperparameter tuning skills, and speech project practice within 2 weeks.

---

## Day 1: PyTorch Basics + Environment Setup (Local Mac)

**Goal**: Understand tensors, models, loss, optimizer

* Video: freeCodeCamp - Deep Learning with PyTorch (first 2 hours intro chapters)
* Tasks:
  * Setup environment (PyTorch, Jupyter Notebook, matplotlib)
  * Complete official Hello World: linear regression + single hidden layer neural network
* Output: Save notebook, include loss curves
* Table records: input dimensions, output dimensions, initial loss value, training epochs

---

## Day 2: Training Process + Optimizer & Loss (Local Mac)

**Goal**: Master training loop, loss functions, optimizer

* Video: freeCodeCamp PyTorch training loop chapters (~2h)
* Tasks:
  * Train simple MLP on MNIST
  * Try different losses: CrossEntropy, MSE
  * Try different optimizers: SGD, Adam
* Output: Training curves + accuracy
* Table records: optimizer, learning rate, final loss, accuracy

---

## Day 3: MNIST CNN Project + Activation Function Experiments (Local Mac)

**Goal**: Understand CNN structure and gradients

* Video: YouTube Deeplizard - CNN MNIST (2h)
* Tasks:
  * Build CNN model
  * Compare ReLU, LeakyReLU, Tanh activations
  * Visualize layer outputs and gradient norms
* Output: Training curves + activation function comparison plots
* Table records: activation function, final loss value, accuracy, gradient norm

---

## Day 4: CIFAR10 Project + First Hyperparameter Tuning Experience (Local Mac)

**Goal**: Learn hyperparameter tuning logic

* Video: YouTube Aladdin Persson - CIFAR10 CNN tutorial (2h)
* Tasks:
  * Train CNN/CNN+BatchNorm
  * Adjust batch size, learning rate, Dropout
  * Observe loss/accuracy curve changes
* Output: Training curves + parameter adjustment records
* Table records: batch size, learning rate, dropout, accuracy

---

## Day 5: CIFAR10 Deep Hyperparameter Tuning + Visualization Tools (Local Mac)

**Goal**: Master TensorBoard / wandb visualization

* Video: TensorBoard PyTorch tutorial (1h) + wandb tutorial (1h)
* Tasks:
  * Integrate TensorBoard / wandb, visualize loss/accuracy/gradients
  * Adjust optimizer, scheduler
  * Record training curves for each run
* Output: Visualization Dashboard
* Table records: parameter combinations, accuracy, loss curve screenshots

---

## Day 6: Container Basics + GPU Usage Introduction (School GPU)

**Goal**: Learn to use containers for training on school GPU

* Video tutorials / Documentation:
	* Apptainer official tutorial: https://apptainer.org/docs/
	* Podman basics tutorial: https://podman.io/getting-started/
	* YouTube search: "Singularity / Apptainer PyTorch tutorial"
* Tasks:
  * Learn to launch GPU containers (apptainer exec --nv pytorch.sif bash)
  * Mount local project code
  * Test GPU availability (torch.cuda.is_available())
* Output: Successfully run PyTorch examples (like MNIST) in container
* Table records: container engine, GPU availability, PyTorch version
* Container requirement: ✅ Must practice on school GPU

---

## Day 7: Speech Commands Introduction (Local Mac or GPU Container)

**Goal**: Implement simple speech classification

* Video: Deeplizard Speech Commands tutorial (2h)
* Tasks:
  * Download Google Speech Commands dataset
  * MFCC / Spectrogram preprocessing
  * Build CNN speech classification model
* Output: Training loss/accuracy
* Table records: model structure, loss, accuracy, feature type
* Container requirement: ✅ Can run locally, small GPU container for faster acceleration

---

## Day 8: Speech Commands Hyperparameter Tuning + Visualization (GPU Container)

**Goal**: Practice tuning logic + visualize training

* Video: Aladdin Persson - Speech Commands advanced (1.5h)
* Tasks:
  * Tune learning rate, batch size, activation functions
  * Use TensorBoard / matplotlib for visualization
  * Save each tuning result
* Output: Training curves + best parameter records
* Table records: parameter combinations, final loss value, accuracy, gradient norm
* Container requirement: ✅ GPU container

---

## Day 9: ASR Wav2Vec2 Introduction (GPU Container)

**Goal**: End-to-end speech recognition

* Video: HuggingFace Wav2Vec2 fine-tune tutorial (2h)
* Tasks:
  * Download pre-trained model Wav2Vec2-base
  * Small-scale fine-tune speech-to-text task
  * Observe loss and accuracy changes
* Output: Training curves + transcription results
* Table records: learning rate, batch size, loss, accuracy
* Container requirement: ✅ GPU container

---

## Day 10: ASR Model Hyperparameter Tuning + Results Analysis (GPU Container)

**Goal**: Master ASR tuning + WER visualization

* Video: HuggingFace + PyTorch Lightning ASR tuning tutorial (1h)
* Tasks:
  * Tune optimizer, scheduler, batch size
  * Visualize loss, WER (Word Error Rate)
* Output: Loss training curves + WER comparison table
* Table records: tuning combinations, loss, WER, accuracy
* Container requirement: ✅ GPU container

---

## Day 11: Voice Cloning / Tacotron2 Introduction (GPU Container)

**Goal**: Learn Tacotron2 + WaveGlow speech generation

* Video: Real-Time Voice Cloning tutorial (2h)
* Tasks:
  * Download Real-Time Voice Cloning project
  * Run examples, generate speech
  * Understand Tacotron2 / WaveGlow pipeline
* Output: Synthesized audio files + training observation logs
* Table records: model parameters, audio samples, loss
* Container requirement: ✅ GPU container

---

## Day 12: Voice Cloning Hyperparameter Tuning + Visualization (GPU Container)

**Goal**: Adjust Tacotron2/Encoder parameters

* Video: Real-Time Voice Cloning advanced tutorial (1h)
* Tasks:
  * Tune encoder, decoder, learning rate
  * Visualize gradients, loss, audio quality comparison
* Output: Best synthesized audio + loss curves
* Table records: parameter combinations, loss, gradient norm, audio quality
* Container requirement: ✅ GPU container

---

## Day 13: Speaker Verification Introduction

**Goal**: Speaker identification

* Video: SpeechBrain Speaker Verification tutorial (2h)
* Tasks:
  * Use pre-trained models for speaker identification
  * Generate d-vectors, cosine similarity verification
* Output: Recognition accuracy + loss curves
* Table records: model, loss, accuracy, similarity
* Container requirement: ✅ GPU container

---

## Day 14: Speaker Verification Tuning + Speech Enhancement Basics

**Goal**: Speaker Verification + speech enhancement

* Video: SpeechBrain Enhancement & SV tutorial (1.5h)
* Tasks:
  * Add speech enhancement module
  * Tune optimizer, learning rate, batch size
  * Visualize loss, d-vector distribution
* Output: Before/after enhancement comparison + loss curves
* Table records: parameter combinations, loss, accuracy, enhancement effects
* Container requirement: ✅ GPU container

---

## Day 15: Review + Project Summary + Documentation Organization

* Tasks:
  * Organize all project notebooks + code + tables
  * Output training curves, hyperparameter summary tables, final accuracy for each project
  * Write one-page summary: tuning insights + gradient observations + visualization screenshots
* Output: Final project templates + hyperparameter summary tables + visualization charts

---

### 📂 Appendix: Project Starting Template + Roadmap File Structure Suggestions

```
project_fasttrack/
│
├── day1_mnist_baseline/
│   ├── notebook.ipynb
│   ├── loss_curve.png
│   └── params_table.csv
│
├── day2_cifar10_cnn/
│   ├── notebook.ipynb
│   ├── loss_accuracy_curve.png
│   └── params_table.csv
│
├── day3_speech_commands/
│   ├── notebook.ipynb
│   ├── mfcc_features.npy
│   ├── loss_accuracy_curve.png
│   └── params_table.csv
│
├── day4_wav2vec2_asr/
│   ├── notebook.ipynb
│   ├── fine_tuned_model/
│   └── loss_accuracy_table.csv
│
├── day5_voice_cloning/
│   ├── notebook.ipynb
│   ├── synthesized_audio.wav
│   └── params_table.csv
│
└── day6_speaker_verification/
    ├── notebook.ipynb
    ├── d_vectors.npy
    ├── loss_accuracy_curve.png
    └── params_table.csv
```

---

### 📌 Notes

* 8 hours daily, can be divided into 4×2h or 2×4h blocks to maintain coherent thinking
* Daily must-dos: run code + visualization + table records, learn while doing
* Combine video + hands-on practice, ensure complete coverage from zero to tunable models

## 📌 Video Resources
- [PyTorch Basics - freeCodeCamp](https://www.youtube.com/watch?v=GIsg-ZUy0MY)
- [CNN + MNIST / CIFAR10 - Deeplizard](https://www.youtube.com/watch?v=gG8q2biSfR0)
- [Speech Commands Introduction - Deeplizard](https://mlarchive.com/deep-learning/speech-command-recognition-the-ultimate-guide/)
- [Speech Commands Advanced - Aladdin Persson](https://www.youtube.com/watch?v=Qj4RyX2Gh4s)
- [Wav2Vec2 ASR Fine-tune - HuggingFace](https://huggingface.co/blog/fine-tune-wav2vec2-english)
- [Real-Time Voice Cloning - GitHub + YouTube](https://github.com/CorentinJ/Real-Time-Voice-Cloning)
- [SpeechBrain Speaker Verification & Enhancement](https://speechbrain.readthedocs.io/en/latest/)
- [Apptainer Getting Started Tutorial](https://apptainer.org/docs/)

### 🐳 Container Training Recommended Resources

- [📦 Docker Introduction Tutorial (Bilibili)](https://www.bilibili.com/video/BV1THKyzBER6/?spm_id_from=333.337.search-card.all.click&vd_source=60fc8fe7df9a5d270abe321b54e20a92)
- [🎥 Podman Getting Started (YouTube)](https://www.youtube.com/watch?v=iJe0qzO8EHs)
- [📘 Apptainer Official Documentation](https://apptainer.org/docs/user/latest/)
- [🔧 Conda + Container with Jupyter Tutorial](https://www.bilibili.com/video/BV1Z7411L7dy/?spm_id_from=333.337.search-card.all.click&vd_source=60fc8fe7df9a5d270abe321b54e20a92)

> Usage instructions and container configuration files are uniformly placed in the `containers/` directory. Whether each specific project uses containerization will be noted in the corresponding `notes.md`.

---

## 🔑 GitHub Usage Suggestions

- `.DS_Store`, `__pycache__`, `*.ipynb_checkpoints` and other files are added to `.gitignore` to prevent committing invalid content
- Recommend using **SSH Key** instead of password/PAT for Git commits. Setup instructions:
  - [📘 GitHub SSH Key Tutorial](https://docs.github.com/en/authentication/connecting-to-github-with-ssh)

---

## 🔗 Other Reference Projects & Resource Collections (Updating...

- [PyTorch Official 60-Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
- [Scikit-learn Official Tutorial Collection](https://scikit-learn.org/)
- [Real-time Speech Recognition Project based on faster-whisper (Bilibili)](https://www.bilibili.com/video/BV1fQ4y1j7wb/) 
- [whisper.cpp Most Detailed Installation Tutorial (Bilibili)](https://www.bilibili.com/video/BV19L411v7cq/) 
- [From wav2vec2.0 to HuBERT (Bilibili)](https://www.bilibili.com/video/BV1ea411r7Wg/) 
- [SpeechBrain Open Source Project](https://speechbrain.readthedocs.io/)
- [ESPnet Open Source Project](https://espnet.github.io/espnet/)

---

## 💡 About Me

This repository is maintained by Ming Jin, aiming to improve practical project skills through public documentation and adapt to research/industrial application requirements. Welcome to submit issues/PRs for suggestions and discussions.