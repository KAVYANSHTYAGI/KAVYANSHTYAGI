<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f0c29,50:302b63,100:24243e&height=200&section=header&text=Kavyansh%20Tyagi&fontSize=48&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=ML%20Researcher%20%7C%20NIT%20Kurukshetra&descAlignY=58&descSize=18" />

</div>

<h3 align="center">
Efficient Vision &nbsp;•&nbsp; State Space Models &nbsp;•&nbsp; Event-Based Vision &nbsp;•&nbsp; Machine Unlearning
</h3>

<p align="center">
  <em>Final-year B.Tech (ECE) undergraduate &nbsp;|&nbsp; NIT Kurukshetra, India &nbsp;|&nbsp; Class of 2027</em><br/>
  <em>Collaborating with Sapienza University of Rome and the University of Tübingen</em>
</p>

<p align="center">
  <a href="https://kavyanshtyagi.github.io"><img src="https://img.shields.io/badge/Website-111010?style=for-the-badge&logo=githubpages&logoColor=white"/></a>
  <a href="https://scholar.google.com/citations?user=9Kg8eQUAAAAJ&hl=en"><img src="https://img.shields.io/badge/Google_Scholar-4285F4?style=for-the-badge&logo=googlescholar&logoColor=white"/></a>
  <a href="https://www.linkedin.com/in/kavyansh-tyagi-9b5410264/"><img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white"/></a>
  <a href="mailto:kavyansh.tyagi7@gmail.com"><img src="https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white"/></a>
</p>

---

## 🧭 Research Identity

I build **efficient, robust models** for settings where compute, memory and privacy are real constraints, not afterthoughts. The common thread is models that understand the structure of the data instead of memorizing surface statistics.

- **State space models & hyperspectral vision**: Mamba/SSM architectures that respect the physics of the signal (SCAF, AbSpec-Mamba)
- **3D medical segmentation**: sub-1M-parameter models for memory-constrained clinical hardware (LightMedSeg, RefineFormer3D)
- **Event-based vision**: treating event rate as a feature rather than a nuisance (CEGNN)
- **Machine unlearning & privacy auditing**: showing that "forgotten" classes still leak through a model's spatial and temporal output (ADVR, PACE)

> *"Efficient by design. Principled by necessity."*

---

## 📄 Publications

| Paper | Venue |
|---|---|
| **Beyond the Frame: Redundancy, Not Trajectory, Carries the Privacy Leak in Video Unlearning** (PACE) · Tyagi & Russo · [project page](https://kavyanshtyagi.github.io/PACE-Redundancy-Not-Trajectory-Carries-the-Privacy-Leak-in-Video-Unlearning/) · [code](https://github.com/KAVYANSHTYAGI/PACE-Redundancy-Not-Trajectory-Carries-the-Privacy-Leak-in-Video-Unlearning) | WACV 2027 |
| **Rate as a Feature, Not a Nuisance: Curriculum-FiLM Event GNN** (CEGNN) · Tyagi et al. · [code](https://github.com/kavyanshtyagi/CE-GNN) | ACCV 2026 Workshops (AECV) |
| **LightMedSeg: Lightweight 3D Medical Image Segmentation with Learned Spatial Anchors** · Tyagi, Rathi, Goyal · [arXiv](https://arxiv.org/abs/2603.07228) | CVPR Workshops (ECV) 2026 |
| **RefineFormer3D: Efficient 3D Medical Image Segmentation via Adaptive Multi-Scale Transformer with Cross Attention Fusion** · Tyagi, Rathi, Goyal · [arXiv](https://arxiv.org/abs/2602.16320) | Preprint, 2026 |
| **SCAF: Robust Hyperspectral Classification via Optimal Transport-Based Spectral Grouping** · Tyagi et al. | Preprint, under review |
| **Ghidra-Assisted Static Analysis and Ensemble Learning with Differential-Privacy GANs for Ransomware Detection** · Chaturvedi et al. | ICDAM 2025 |

---

## 🔬 Selected Work

### 🏥 LightMedSeg: 3D medical segmentation in 0.48M parameters
`PyTorch` `3D CNNs` `Spatial Anchoring` `BraTS` `ACDC`

Learned spatial anchors guide the decoder without full-resolution feature maps at every layer. **84.8% Dice on BraTS, 93.5% on ACDC, 0.48M parameters, 14 GFLOPs.** → [arXiv](https://arxiv.org/abs/2603.07228)

### 🧪 ADVR & PACE: privacy leakage in class unlearning
`PyTorch` `Adversarial training` `Semantic segmentation` `Video` `Mamba attackers`

Segmentation unlearning that relabels a class leaves a class-shaped hole in the output. **ADVR** removes the silhouette itself and is the only method leak-safe on both output and feature surfaces (ADE20K, Cityscapes). **PACE** extends this to video: averaging outputs over a ~2 s clip recovers the forgotten class from every published baseline, and the signal comes from cross-frame redundancy, not trajectory. → [ADVR](https://github.com/KAVYANSHTYAGI/advr-unlearning) · [PACE](https://github.com/KAVYANSHTYAGI/PACE-Redundancy-Not-Trajectory-Carries-the-Privacy-Leak-in-Video-Unlearning)

### 🌐 CEGNN: rate as a feature in event-camera GNNs
`PyTorch Geometric` `GNNs` `Event cameras` `FiLM`

The log event rate of each recording drives identity-initialised FiLM layers, trained with a two-stage rate curriculum: **+14.0 points on DVS128 Gesture, +6.3 on N-Caltech101**, and +34 points under synthetic speed shifts, for 49K extra parameters. Negative results documented. → [Code](https://github.com/KAVYANSHTYAGI/CE-GNN)

### 🌈 SCAF: robust hyperspectral classification
`Mamba` `Optimal transport` `Sinkhorn` `Hyperspectral`

Differentiable optimal-transport grouping of spectral bands with per-group Mamba processors turns spectral redundancy into graceful degradation: **+11.6% accuracy under 20–30% band dropout** against Transformer baselines.

### 🛰️ In progress
- **AbSpec-Mamba / VolMamba v2**: abundance-conditioned selective state transitions for physically grounded hyperspectral super-resolution (with Sapienza University of Rome)
- **S6MOD**: regret-driven adaptive routing for mixture-of-experts under distribution shift
- **Selective Holographic State (SHS)**: sub-quadratic attention alternative based on low-rank holographic projections

---

## 🏗️ Building

- **[Reflara](https://github.com/Reflara-ai/reflara-oss)**: replay-native execution-intelligence platform for autonomous AI agents. Deterministic record and replay of agent trajectories, written in Rust.
- **[Qten.ai](https://qten.ai)**: AI engineering intern, building agentic orchestration for grounded conversational content generation.
- **SIGPLAN-M**: operations team member, organizing international mentoring for programming-languages researchers.

---

## 🌍 Open Source

| Project | Merged PRs | Focus |
|---|---|---|
| **[Lightning AI](https://github.com/Lightning-AI/lightning)** | 16 | Model compilation pipelines, argument-parsing edge cases, distributed training stability |
| **[OpenCV](https://github.com/opencv/opencv)** | 9 | Bug fixes and API robustness in core computer-vision pipelines |

---

## 🛠️ Technical Stack

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/C++-00599C?style=for-the-badge&logo=cplusplus&logoColor=white" />
  <img src="https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white" />
  <img src="https://img.shields.io/badge/Rust-000000?style=for-the-badge&logo=rust&logoColor=white" />
  <img src="https://img.shields.io/badge/Bash-121011?style=for-the-badge&logo=gnubash&logoColor=white" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch_Lightning-792EE5?style=for-the-badge&logo=lightning&logoColor=white" />
  <img src="https://img.shields.io/badge/JAX-3EBA77?style=for-the-badge&logo=jax&logoColor=black" />
  <img src="https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=opencv&logoColor=white" />
  <img src="https://img.shields.io/badge/HuggingFace-FFD21F?style=for-the-badge&logo=huggingface&logoColor=black" />
  <img src="https://img.shields.io/badge/torch_geometric-9933cc?style=for-the-badge" />
  <img src="https://img.shields.io/badge/timm-0099FF?style=for-the-badge" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=for-the-badge&logo=weightsandbiases&logoColor=black" />
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" />
  <img src="https://img.shields.io/badge/GitHub_Actions-2088FF?style=for-the-badge&logo=github-actions&logoColor=white" />
  <img src="https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black" />
</p>

---

## 📊 GitHub Statistics

<p align="center">
  <img width="48%" src="https://github-readme-stats.vercel.app/api?username=kavyanshtyagi&show_icons=true&theme=tokyonight&hide_border=true&count_private=true" />
  <img width="48%" src="https://streak-stats.demolab.com/?user=kavyanshtyagi&theme=tokyonight&hide_border=true" />
</p>

<p align="center">
  <img src="https://github-readme-activity-graph.vercel.app/graph?username=kavyanshtyagi&theme=tokyo-night&hide_border=true&area=true" />
</p>

---

## 🤝 Open To

- 📝 **Research collaborations**: efficient vision, state space models, event-based vision, machine unlearning and privacy auditing
- 🎓 **Graduate research opportunities** (2027 onward)
- 💬 **Discussions**: architecture design, SSMs, unlearning evaluation, CUDA optimization

> 📩 **kavyansh.tyagi7@gmail.com**

---

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:24243e,50:302b63,100:0f0c29&height=100&section=footer"/>

*"Efficient by design. Principled by necessity."*
</div>
