# In-Context Brush: Zero-shot Customized Subject Insertion with Context-Aware Latent Space Manipulation [ACM SIGGRAPH ASIA 2025]

> **In-Context Brush: Zero-shot Customized Subject Insertion with Context-Aware Latent Space Manipulation**<br>
> Yu Xu<sup>1,2</sup>, Fan Tang<sup>1</sup>, You Wu<sup>1</sup>, Lin Gao<sup>1</sup>, Oliver Deussen<sup>3</sup>, Hongbin Yan<sup>2</sup>, Jintao Li<sup>1</sup>, Juan Cao<sup>1</sup>, Tong-Yee Lee<sup>4</sup> <br>
> <sup>1</sup>Institute of Computing Technology, Chinese Academy of Sciences, <sup>2</sup>University of Chinese Academy of Sciences, <sup>3</sup>University of Konstanz, <sup>4</sup>National Cheng Kung University

![](assets/teaser.jpeg)

<a href='https://arxiv.org/abs/2505.20271'><img src='https://img.shields.io/badge/ArXiv-2505.20271-red'></a> 

>**Abstract**: <br>
>Recent advances in diffusion models have enhanced multimodal-guided visual generation, enabling customized subject insertion that seamlessly "brushes" user-specified objects into a given image guided by textual prompts. However, existing methods often struggle to insert customized subjects with high fidelity and align results with the user's intent through textual prompts. In this work, we propose "In-Context Brush", a zero-shot framework for customized subject insertion by reformulating the task within the paradigm of in-context learning. Without loss of generality, we formulate the object image and the textual prompts as cross-modal demonstrations, and the target image with the masked region as the query. The goal is to inpaint the target image with the subject aligning textual prompts without model tuning. Building upon a pretrained MMDiT-based inpainting network, we perform test-time enhancement via dual-level latent space manipulation: intra-head "latent feature shifting" within each attention head that dynamically shifts attention outputs to reflect the desired subject semantics and inter-head "attention reweighting" across different heads that amplifies prompt controllability through differential attention prioritization. Extensive experiments and applications demonstrate that our approach achieves superior identity preservation, text alignment, and image quality compared to existing state-of-the-art methods, without requiring dedicated training or additional data collection.

## More of our results
![](assets/more_results.png)
