# 🧠 ML Speech Learning ｜语音+机器学习 打卡学习记录

**Language**: [English](README_EN.md) | 中文

本仓库用于系统记录从零学习语音方向机器学习的全过程，包括基础课程、实战项目、容器训练、每日总结打卡等内容。目标是通过 2–3 周高强度学习，完成从入门到实践的过渡，并为未来的研究或实习积累项目经验和代码成果。
初始的学习计划和资源来自和ChatGPT的讨论，所以我也不知道15天后的结果。这也可以作为一个使用ChatCPT学习的实验，让我们拭目以待。如果谁想按照这个计划一起学习，请告诉我，让我知道我不是一个人。。❤️😭💪🏻

---

# Machine Learning & Speech Project Learning

## 🎯 学习目标
- 掌握 PyTorch 基础
- 能独立训练 CNN、语音分类、ASR、Voice Cloning、Speaker Verification
- 理解梯度、loss、调参和可视化训练过程
- 学会在 Mac 和 GPU 容器环境下训练

## 📅 学习计划
- 15 天快速上手计划
- 每天 8 小时，包括视频学习 + 代码实践 + 调参记录
- 项目列表：
  1. PyTorch 基础
  2. CNN 图像分类 (MNIST / CIFAR10)
  3. 语音分类 (Speech Commands)
  4. ASR Wav2Vec2
  5. Voice Cloning
  6. Speaker Verification + Enhancement

## 📁 仓库结构
- 每天一个文件夹，包含 notebook / 调参表格 / 学习笔记 / 可视化图
- Day15 汇总总结经验与可视化结果
```
ml_speech_learning/
├── README.md              # 项目主页说明
├── Day1/                  # 每天的学习任务记录
│   ├── notebook.ipynb     # 学习代码 + 实验记录
│   └── notes.md           # 总结 / 反思 / 链接 / 提问
├── Day2/ ~ Day15/         # 其他日任务（后续添加）
├── utils/                 # 公共脚本函数，如特征提取
├── containers/            # Dockerfile / Apptainer 配置文件
├── .gitignore             # 忽略不必要文件（如.DS_Store）
└── LICENSE                # 项目许可（MIT）
```

## 🧑‍💻 每日学习方式

每个 `DayX/` 文件夹建议包含：

- `notebook.ipynb`：对应学习主题的代码（示例见 Day1）
- Day1 示例目录模板:
```
Day1_PyTorch_Basics/
├── 01_notebook.ipynb # Jupyter Notebook 文件
├── 02_loss_curve.png # 训练曲线截图
├── 03_params_table.csv # 调参记录表格
└── 04_notes.md # 学习笔记（概念 + 遇到问题 + 解决方案）
```

- `notes.md`：当日学习总结、遇到问题、链接、反思等
- Day1 示例目录模板:
- Notebook 文件命名规则：
```
DayX_ProjectName_Description.ipynb
DayX_ProjectName_params.csv
DayX_ProjectName_loss.png
DayX_ProjectName_notes.md
```
## ⚡ 使用说明
- 可直接打开 Jupyter Notebook 运行
- 调参结果和图表可在 `*_curve.png` 或 `params_table.csv` 中查看
- 每日完成后提交更新：

```bash
git add .
git commit -m "Add DayX study files"
git push origin main
```

---

## ✏️ 每日计划（每天 8 小时）

目标：2 周内快速掌握 PyTorch、调参能力和语音项目实战。

---

## Day 1：PyTorch 基础 + 环境搭建（本地 Mac）

**目标**：理解张量、模型、loss、optimizer

* 视频：freeCodeCamp - Deep Learning with PyTorch（前 2 小时入门章节）
* 任务：

  * 搭建环境（PyTorch, Jupyter Notebook, matplotlib）
  * 完成官方 Hello World：线性回归 + 单隐藏层神经网络
* 输出：Notebook 保存，写明 loss 曲线
* 表格记录：输入维度、输出维度、loss 初始值、训练轮次

---

## Day 2：训练流程 + optimizer 与 loss（本地 Mac）

**目标**：掌握训练循环、loss 函数、optimizer

* 视频：freeCodeCamp PyTorch 训练循环章节（约 2h）
* 任务：

  * 用 MNIST 训练简单 MLP
  * 尝试不同 loss: CrossEntropy, MSE
  * 尝试不同优化器: SGD, Adam
* 输出：训练曲线 + 精度
* 表格记录：optimizer, learning rate, final loss, accuracy

---

## Day 3：MNIST CNN 项目 + 激活函数实验（本地 Mac）

**目标**：理解 CNN 结构与梯度

* 视频：YouTube Deeplizard - CNN MNIST (2h)
* 任务：

  * 建立 CNN 模型
  * 比较 ReLU, LeakyReLU, Tanh 激活
  * 可视化每层输出和梯度 norm
* 输出：训练曲线 + 各激活函数对比图
* 表格记录：激活函数, loss 最终值, accuracy, 梯度 norm

---

## Day 4：CIFAR10 项目 + 调参初体验（本地 Mac）

**目标**：学习调参逻辑

* 视频：YouTube Aladdin Persson - CIFAR10 CNN tutorial (2h)
* 任务：

  * 训练 CNN/CNN+BatchNorm
  * 改 batch size、学习率、Dropout
  * 观察 loss/accuracy 曲线变化
* 输出：训练曲线 + 参数调整记录
* 表格记录：batch size, learning rate, dropout, accuracy

---

## Day 5：CIFAR10 深度调参 + 可视化工具（本地 Mac）

**目标**：掌握 TensorBoard / wandb 可视化

* 视频：TensorBoard PyTorch 教程（1h）+ wandb 教程（1h）
* 任务：

  * 整合 TensorBoard / wandb，可视化 loss/accuracy/梯度
  * 调整 optimizer, scheduler
  * 记录各轮训练曲线
* 输出：可视化 Dashboard
* 表格记录：参数组合, accuracy, loss 曲线截图

---

## Day 6：容器基础 + GPU 使用入门（学校 GPU）

**目标**：学会在学校 GPU 上使用容器训练

* 视频教程 / 文档：
	* Apptainer 官方教程：https://apptainer.org/docs/
	* Podman 基础教程：https://podman.io/getting-started/
	* YouTube 搜索：“Singularity / Apptainer PyTorch tutorial”
* 任务：

  * 学会启动 GPU 容器 (apptainer exec --nv pytorch.sif bash)
  * 挂载本地项目代码
  * 测试 GPU 是否可用 (torch.cuda.is_available())
* 输出：成功在容器中运行 PyTorch 示例（如 MNIST）
* 表格记录：容器引擎、GPU 是否可用、PyTorch 版本
* 容器需求：✅ 必须在学校 GPU 上实践

---

## Day 7：Speech Commands 入门（本地 Mac 或 GPU 容器）

**目标**：实现简单语音分类

* 视频：Deeplizard Speech Commands tutorial（2h）
* 任务：

  * 下载 Google Speech Commands 数据集
  * MFCC / Spectrogram 预处理
  * 建立 CNN 语音分类模型
* 输出：训练 loss/accuracy
* 表格记录：模型结构, loss, accuracy, 特征类型
* 容器需求：✅ 本地可以跑，小 GPU 容器加速更快

---

## Day 8：Speech Commands 调参 + 可视化（GPU 容器）

**目标**：练调参逻辑 + 可视化训练

* 视频：Aladdin Persson - Speech Commands advanced (1.5h)
* 任务：

  * 调学习率、batch size、激活函数
  * 使用 TensorBoard / matplotlib 可视化
  * 保存每次调参结果
* 输出：训练曲线 + 最佳参数记录
* 表格记录：参数组合, loss 最终值, accuracy, gradient norm
* 容器需求：✅ GPU 容器

---

## Day 9：ASR Wav2Vec2 入门（GPU 容器)

**目标**：端到端语音识别

* 视频：HuggingFace Wav2Vec2 fine-tune 教程（2h）
* 任务：

  * 下载预训练模型 Wav2Vec2-base
  * 小规模 fine-tune 语音转文本任务
  * 观察 loss 和精度变化
* 输出：训练曲线 + 转录效果
* 表格记录：learning rate, batch size, loss, accuracy
* 容器需求：✅ GPU 容器

---

## Day 10：ASR 模型调参 + 结果分析（GPU 容器）

**目标**：掌握 ASR 调参 + WER 可视化

* 视频：HuggingFace + PyTorch Lightning ASR 调参教程（1h）
* 任务：

  * 调 optimizer, scheduler, batch size
  * 可视化 loss、WER（Word Error Rate）
* 输出：loss 训练曲线 + WER 对比表格
* 表格记录：调参组合, loss, WER, accuracy
* 容器需求：✅ GPU 容器

---

## Day 11：Voice Cloning / Tacotron2 入门（GPU 容器）

**目标**：学会 Tacotron2 + WaveGlow 生成语音

* 视频：Real-Time Voice Cloning 教程（2h）
* 任务：

  * 下载 Real-Time Voice Cloning 项目
  * 运行示例，生成语音
  * 理解 Tacotron2 / WaveGlow 流程
* 输出：合成音频文件 + 训练观察日志
* 表格记录：模型参数, 音频示例, loss
* 容器需求：✅ GPU 容器

---

## Day 12：Voice Cloning 调参 + 可视化（GPU 容器）

**目标**：调整 Tacotron2/Encoder 参数

* 视频：Real-Time Voice Cloning 高级教程（1h）
* 任务：

  * 调 encoder, decoder, learning rate
  * 可视化梯度、loss, 音频效果对比
* 输出：最佳合成音频 + loss曲线
* 表格记录：参数组合, loss, 梯度 norm, 音频质量
* 容器需求：✅ GPU 容器

---

## Day 13：Speaker Verification 入门

**目标**：说话人识别

* 视频：SpeechBrain Speaker Verification 教程（2h）
* 任务：

  * 使用预训练模型做说话人识别
  * 生成 d-vector, cosine similarity 验证
* 输出：识别准确率 + loss曲线
* 表格记录：模型, loss, accuracy, similarity
* 容器需求：✅ GPU 容器

---

## Day 14：Speaker Verification 调参 + 语音增强初步

**目标**：Speaker Verification + 语音增强

* 视频：SpeechBrain Enhancement & SV tutorial（1.5h）
* 任务：

  * 添加语音增强模块
  * 调参 optimizer, learning rate, batch size
  * 可视化 loss, d-vector 分布
* 输出：增强前后效果对比 + loss曲线
* 表格记录：参数组合, loss, accuracy,增强效果
* 容器需求：✅ GPU 容器

---

## Day 15：复盘 + 项目总结 + 文档整理

**时间**：8h

* 任务：

  * 整理所有项目 notebook + 代码 + 表格
  * 输出每个项目的训练曲线、调参总结表格、最终精度
  * 写一页总结：调参心得 + 梯度观察 + 可视化截图
* 输出：最终项目模板 + 调参总结表格 + 可视化图表

---

### 📂 附件：项目起始模板 + 路线图文件结构建议

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

### 📌 备注

* 每天 8 小时，可分 4×2h 或 2×4h block，保证思路连贯
* 每天必须完成：代码运行 + 可视化 + 表格记录，边做边学
* 视频 + 实操结合，确保从零到可调模型全覆盖

## 📌 视频资源
- [PyTorch 基础 - freeCodeCamp](https://www.youtube.com/watch?v=GIsg-ZUy0MY)
- [CNN + MNIST / CIFAR10 - Deeplizard](https://www.youtube.com/watch?v=gG8q2biSfR0)
- [Speech Commands 入门 - Deeplizard](https://mlarchive.com/deep-learning/speech-command-recognition-the-ultimate-guide/)
- [Speech Commands 高级 - Aladdin Persson](https://www.youtube.com/watch?v=Qj4RyX2Gh4s)
- [Wav2Vec2 ASR Fine-tune - HuggingFace](https://huggingface.co/blog/fine-tune-wav2vec2-english)
- [Real-Time Voice Cloning - GitHub + YouTube](https://github.com/CorentinJ/Real-Time-Voice-Cloning)
- [SpeechBrain Speaker Verification & Enhancement](https://speechbrain.readthedocs.io/en/latest/)
- [Apptainer 入门教程](https://apptainer.org/docs/)

### 🐳 容器训练推荐资源

- [📦 Docker 入门教程（B站）](https://www.bilibili.com/video/BV1THKyzBER6/?spm_id_from=333.337.search-card.all.click&vd_source=60fc8fe7df9a5d270abe321b54e20a92)
- [🎥 Podman 使用入门（YouTube）](https://www.youtube.com/watch?v=iJe0qzO8EHs)
- [📘 Apptainer 官方文档](https://apptainer.org/docs/user/latest/)
- [🔧 Conda + 容器搭配 Jupyter 教程](https://www.bilibili.com/video/BV1Z7411L7dy/?spm_id_from=333.337.search-card.all.click&vd_source=60fc8fe7df9a5d270abe321b54e20a92)

> 使用说明与容器配置文件统一放在 `containers/` 目录下，具体每个项目是否容器化将在对应 `notes.md` 中注明。

---

## 🔑 GitHub 使用建议

- `.DS_Store`、`__pycache__`、`*.ipynb_checkpoints` 等文件已加入 `.gitignore`，防止提交无效内容
- 推荐使用 **SSH Key** 而非密码 / PAT 进行 Git 提交，设置方式见：
  - [📘 GitHub SSH Key 教程](https://docs.github.com/en/authentication/connecting-to-github-with-ssh)

---

## 🔗 其他参考项目与资料合集（更新中...

- [PyTorch 官方 60 分钟 Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
- [Scikit-learn 官方教程合集](https://scikitlearn.com.cn/)
- [基于 faster‑whisper 实现实时语音识别项目（B站）](https://www.bilibili.com/video/BV1fQ4y1j7wb/) 
- [基于 faster‑whisper 实现实时语音识别项目（B站）](https://www.bilibili.com/video/BV1fQ4y1j7wb/) 
- [whisper.cpp 最详细安装教程（B站）](https://www.bilibili.com/video/BV19L411v7cq/) 
- [从 wav2vec2.0 到 HuBERT（B站）](https://www.bilibili.com/video/BV1ea411r7Wg/) 
- [SpeechBrain 开源项目](https://speechbrain.readthedocs.io/)
- [ESPnet 开源项目](https://espnet.github.io/espnet/)

---

## 💡 关于我

此仓库由 Ming Jin 维护，旨在通过公开记录提升项目实战能力，并适应科研/工业应用要求。欢迎 issue/PR 建议与交流。
