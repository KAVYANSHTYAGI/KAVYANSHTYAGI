<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f0c29,50:302b63,100:24243e&height=200&section=header&text=Kavyansh%20Tyagi&fontSize=48&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=AI%20Researcher%20%7C%20NIT%20Kurukshetra&descAlignY=58&descSize=18" />

</div>

<h3 align="center">
3D Medical Segmentation &nbsp;•&nbsp; Efficient Vision Architectures &nbsp;•&nbsp; Event-Based Vision
</h3>

<p align="center">
  <em>3rd Year Undergraduate Researcher &nbsp;|&nbsp; NIT Kurukshetra, India</em><br/>
  <em>Published in Springer & Elsevier &nbsp;|&nbsp; Targeting CVPR · MICCAI · NeurIPS · ICLR</em>
</p>

<p align="center">
  <a href="mailto:kavyansh.tyagi7@gmail.com"><img src="https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white"/></a>
  <a href="https://www.linkedin.com/in/kavyansh-tyagi-9b5410264/"><img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white"/></a>
  <a href="https://github.com/kavyanshtyagi"><img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white"/></a>
</p>

---

## 🧭 Research Identity

I design **efficient deep learning architectures** for tasks where compute and memory are real constraints — not afterthoughts.

My research centers on three threads:

- **3D Medical Image Segmentation** — building lightweight transformer-based models (RefineFormer3D, LightMedSeg) that match or exceed SOTA on BraTS & Synapse benchmarks while dramatically cutting GFLOPs and parameters
- **Efficient Attention & State Mechanisms** — designing novel sub-quadratic attention alternatives (Selective Holographic State) that trade full-rank attention for structured, hardware-friendly representations without sacrificing expressivity
- **Event-Based Vision & Sparse Graph Learning** — leveraging the asynchronous, sparse nature of event camera data through GNN-based representations (SCAF) that process spatiotemporal structure without dense frame reconstruction

My work is driven by a single question: *how do you get the most out of a model that's small enough to actually deploy?*

> *"Efficient by design. Principled by necessity."*

---

## 🔬 Research Projects

---

### 🧠 RefineFormer3D — Efficient 3D Medical Segmentation
`PyTorch` `CUDA` `Vision Transformers` `3D Attention` `BraTS2023` `Synapse`

**Problem:** Transformer-based 3D segmentation models (SwinUNETR, nnFormer, TransBTS) deliver strong accuracy but at prohibitive compute — limiting real-world clinical deployment.

**Approach:** A lightweight encoder–decoder with anatomical priors injected as learned spatial constraints, paired with a coarse-to-fine refinement module. Memory-efficient 3D attention via custom CUDA kernels. Ablations across attention variants, prior injection depth, and decoder topology.

**Results:**
- Outperforms published SOTA on **BraTS2023** and **Synapse** benchmarks
- Significant reduction in GFLOPs and parameter count vs. transformer baselines
- Full ablation suite validating each architectural decision

**Status:** Manuscript in preparation → **CVPR 2026 Workshop / MICCAI 2026**

---

### 🏥 LightMedSeg — Ultra-Lightweight 3D Segmentation with Learned Spatial Anchors
`PyTorch` `3D CNNs` `Spatial Anchoring` `Medical Imaging` `BraTS` `Synapse`

**Problem:** Even "lightweight" segmentation models assume GPU-class hardware. There is no strong baseline for truly resource-constrained 3D medical segmentation (mobile, edge, embedded clinical devices).

**Approach:** Introduces **learned spatial anchors** — a small set of trainable volumetric reference points that guide the decoder without requiring full-resolution feature maps at every layer. Eliminates skip connection overhead while preserving anatomical localization.

**Contribution:** Achieves competitive Dice scores at a fraction of the memory footprint of existing lightweight baselines. Evaluated on BraTS & Synapse with full FLOPs/parameter/latency profiling.

**Status:** Architecture finalized → benchmarking phase → targeting **MICCAI 2026**

---

### ⚡ SHS — Selective Holographic State for Efficient Attention
`PyTorch` `Custom CUDA` `State Space Models` `Attention Mechanisms` `Sub-quadratic Complexity`

**Problem:** Standard softmax attention scales as O(N²) in sequence length, making it impractical for high-resolution volumetric inputs (3D scans, long event streams). SSM alternatives (Mamba, S6) sacrifice expressivity for efficiency.

**Approach:** **Selective Holographic State (SHS)** — a novel mechanism that represents attention as a low-rank holographic projection over a structured state matrix. Selectively retains interaction patterns based on input-conditioned gating, achieving sub-quadratic complexity without losing global context modeling.

**Contribution:** Designed as a drop-in replacement for self-attention in vision transformer backbones; evaluated on classification, segmentation, and sequence modeling benchmarks.

**Status:** Core implementation complete → ablation and benchmarking → targeting **ICLR / NeurIPS 2026**

---

### 🌐 SCAF — Sparse Graph Learning for Event-Based Vision
`PyTorch Geometric` `GNNs` `Event Cameras` `Sparse Graphs` `Asynchronous Vision`

**Problem:** Event cameras produce asynchronous, sparse spike streams that dense CNN/transformer pipelines cannot efficiently process — they either reconstruct frames (losing temporal precision) or apply attention to padded dense tensors (wasting compute on empty regions).

**Approach:** **SCAF (Sparse-to-Dense Causal Attention Framework)** models event streams directly as dynamic sparse graphs. Nodes represent active pixels; edges encode spatiotemporal proximity. A novel **temporal grouping mechanism** clusters nodes by activity burst patterns to prevent neighborhood explosion during message passing.

**Contribution:** Processes raw event streams without frame reconstruction; +15% over TGN baselines on large-scale temporal graph benchmarks (Wikipedia dataset, 5M+ nodes).

**Status:** Validation phase → targeting **CVPR 2026 / NeurIPS 2026**

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

**Hardware:** RTX 5080 (16GB VRAM) · 64GB RAM · CUDA 12.8 · PyTorch 2.9+

---

## 📄 Publications

- 📘 **[Paper Title]** — *Journal Name*, Springer, 2025. \[[DOI]()\]
- 📗 **[Paper Title]** — *Journal Name*, Elsevier, 2025. \[[DOI]()\]

> *Manuscripts in preparation: CVPR 2026 Workshop · MICCAI 2026 · NeurIPS 2026*

---

## 🌍 Open Source Contributions

| Project | Contribution |
|---|---|
| **PyTorch Lightning** | [Add your PR title + link here] |
| **OpenCV** | [Add your PR title + link here] |

---

## 📊 GitHub Statistics

<p align="center">
  <img width="48%" src="https://github-readme-stats.vercel.app/api?username=kavyanshtyagi&show_icons=true&theme=tokyonight&hide_border=true&count_private=true" />
  <img width="48%" src="https://github-readme-streak-stats.herokuapp.com/?user=kavyanshtyagi&theme=tokyonight&hide_border=true" />
</p>

<p align="center">
  <img src="https://github-readme-activity-graph.vercel.app/graph?username=kavyanshtyagi&theme=tokyo-night&hide_border=true&area=true" />
</p>

---

## 🤝 Currently Open To

- 🔬 **Research Internships (Summer 2026)** — University labs, AI research institutes, industrial research groups
- 📝 **Research Collaborations** — 3D medical imaging, efficient attention, event-based vision, sparse GNNs
- 💬 **Discussions** — Architecture design, sub-quadratic attention, state space models, CUDA optimization

> 📩 **kavyansh.tyagi7@gmail.com**

---

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:24243e,50:302b63,100:0f0c29&height=100&section=footer"/>

*"Efficient by design. Principled by necessity."*
</div>
