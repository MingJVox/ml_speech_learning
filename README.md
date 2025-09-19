# 🧠 ML Speech Learning ｜语音机器学习学习记录仓库

本仓库用于系统记录从零学习语音方向机器学习的全过程，包括基础课程、实战项目、容器训练、每日总结打卡等内容。目标是通过 2–3 周高强度学习，完成从入门到实践的过渡，并为未来的研究或实习积累项目经验和代码成果。

---

## 🎯 学习目标

- 快速掌握 PyTorch、Numpy、Sklearn 等基础工具
- 入门语音信号处理（特征提取、声纹识别、语音识别）
- 学会使用容器工具（Podman / Apptainer）完成模型训练
- 每天坚持学习 & 代码实操，记录成长过程

---

## 🗂️ 仓库结构说明

```
ml_speech_learning/
├── README.md              # 项目主页说明
├── Day1/                  # 每天的学习任务记录
│   ├── notebook.ipynb     # 学习代码 + 实验记录
│   └── notes.md           # 总结 / 反思 / 链接 / 提问
├── Day2/ ~ Day14/         # 其他日任务（后续添加）
├── utils/                 # 公共脚本函数，如特征提取
├── containers/            # Dockerfile / Apptainer 配置文件
├── .gitignore             # 忽略不必要文件（如.DS_Store）
└── LICENSE                # 项目许可（MIT）
```

---

## 🧑‍💻 每日学习方式

每个 `DayX/` 文件夹建议包含：

- `notebook.ipynb`：对应学习主题的代码（示例见 Day1）
- `notes.md`：当日学习总结、遇到问题、链接、反思等

每日完成后提交更新：

```bash
git add .
git commit -m "Add DayX study files"
git push origin main
```

---

## 🧱 学习模块与容器使用计划

| 模块名称             | 内容说明                                     | 是否使用容器 |
|----------------------|----------------------------------------------|--------------|
| Python & Numpy基础   | 快速打通数组操作与基础函数                  | 否           |
| Sklearn入门          | 掌握经典机器学习流程                       | 否           |
| PyTorch入门+实战     | 神经网络、训练流程、模型封装               | 否           |
| 语音特征处理         | 提取 MFCC、Mel谱图等特征                   | 否           |
| 声纹识别项目         | 音频分类或说话人验证入门                   | 否           |
| 语音识别项目         | 使用 Wav2Vec2 / Whisper 跑通 ASR            | ✅ 推荐容器  |
| 容器化训练（重点）   | 学会用 Podman / Apptainer 进行训练复现     | ✅ 必须       |
| 调参与日志记录       | 日志配置、超参数调试、容器内自动保存等    | ✅ 推荐容器  |

---

## 🐳 容器训练推荐资源

- [📦 Docker 入门教程（B站）](https://www.bilibili.com/video/BV1mK411W7kC)
- [🎥 Podman 使用入门（YouTube）](https://www.youtube.com/watch?v=9wlGDEg0j1A)
- [📘 Apptainer 官方文档](https://docs.apptainer.org/)
- [🔧 Conda + 容器搭配 Jupyter 教程](https://www.bilibili.com/video/BV1cP411v7VU)

> 使用说明与容器配置文件统一放在 `containers/` 目录下，具体每个项目是否容器化将在对应 `notes.md` 中注明。

---

## 🔑 GitHub 使用建议

- `.DS_Store`、`__pycache__`、`*.ipynb_checkpoints` 等文件已加入 `.gitignore`，防止提交无效内容
- 推荐使用 **SSH Key** 而非密码 / PAT 进行 Git 提交，设置方式见：
  - [📘 GitHub SSH Key 教程](https://docs.github.com/en/authentication/connecting-to-github-with-ssh)

---

## 🔗 参考项目与资料合集

- [PyTorch 官方 60 分钟 Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
- [Scikit-learn 官方教程合集](https://scikit-learn.org/stable/tutorial/index.html)
- [Wav2Vec2 + Whisper 实战项目（B站）](https://www.bilibili.com/video/BV1Bb4y1c7fG)
- [SpeechBrain 开源项目](https://speechbrain.readthedocs.io/)
- [ESPnet 开源项目](https://espnet.github.io/espnet/)

---

## 💡 关于我

此仓库由 Ming Jin 维护，旨在通过公开记录提升项目实战能力，并适应科研/工业应用要求。欢迎 issue/PR 建议与交流。
