# Explainable Text-to-Image Generation with SHAP and Stable Diffusion

## Overview
This project develops a generative AI system that creates images from text prompts using Stable Diffusion and explains the contribution of each prompt word (e.g., "red" in "red car") to the generated image using SHAP (SHapley Additive exPlanations). Built in Python with Hugging Face Diffusers and SHAP, the system processes 50 prompts from the `nlphuji/coco_captions` dataset, generates high-quality images, and produces visualizations (bar plots, heatmaps) to highlight token importance.

## Use Case
The project supports **ethical AI in content moderation** by detecting biases in generated images. For example, SHAP can reveal if prompt words like "professional" disproportionately influence gender or racial features in outputs, enabling developers to adjust prompts for fairer representations. This addresses real-world challenges in media, advertising, and social platforms, ensuring trustworthy AI deployment.

## Methodology
The project follows these key steps:
1. **Data Preparation**: Loaded 50 diverse text prompts (3-15 words) from the `nlphuji/coco_captions` dataset (validation split) using Hugging Face Datasets, ensuring suitability for text-to-image generation.
2. **Stable Diffusion Setup**: Initialized the `stabilityai/stable-diffusion-2-1` model with Hugging Face Diffusers, optimized for GPU (Google Colab T4) using `torch.float16` and model CPU offloading for memory efficiency.
3. **Image Generation**: Generated 512x512 images for 10 prompts with 50 inference steps and a guidance scale of 7.5, ensuring high-quality outputs.
4. **Fine-Tuning Configuration**: Prepared a LoRA (Low-Rank Adaptation) setup for potential fine-tuning, targeting U-Net modules (`to_k`, `to_q`, `to_v`, `to_out.0`) to enhance prompt-specific performance.
5. **SHAP Explanation**: Developed a custom `StableDiffusionExplainer` class to compute SHAP values for CLIP text embeddings, explaining how prompt tokens influence image features (e.g., RGB means). Used a text masker to analyze token contributions.
6. **Visualization**: Created comprehensive visualizations for each result, including:
   - Generated image display.
   - Bar plots showing SHAP values per token (red for negative, blue for positive contributions).
   - Heatmaps illustrating token contribution intensity.
7. **Analysis**: Summarized results with statistics on total prompts processed and top influential tokens (e.g., "red" with high SHAP values), highlighting key drivers of image generation.

## Results
- **Generated and Explained 10 Images**: Successfully created 512x512 images for 10 COCO captions (e.g., "a red sports car on a mountain road") and explained token contributions with SHAP.
- **Clear Visualizations**: Produced bar plots and heatmaps for each prompt, visually demonstrating token importance (e.g., "red" driving car color).
- **Robust Analysis**: Identified top influential tokens across prompts, with average SHAP values quantifying their impact (e.g., "sunset" with 0.25 importance).
- **Efficient Execution**: Ran on Google Colab’s T4 GPU (16 GB), handling memory constraints with `torch.float16` and CPU offloading, completing in ~2 hours for 10 prompts.

## Setup Instructions
To run the project locally or in Google Colab:
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/Explainable-Text-to-Image-SHAP.git
   cd Explainable-Text-to-Image-SHAP
   ```
2. **Install Dependencies**:
   ```bash
   pip install diffusers transformers accelerate torch torchvision datasets
   pip install shap matplotlib seaborn plotly
   pip install peft
   ```
3. **Run the Code**:
   - Open `Explainable_Text_to_Image_SHAP.ipynb` in Google Colab with GPU enabled (`Runtime` > `Change runtime type` > `GPU`).
   - Execute all cells sequentially to:
     - Load the `nlphuji/coco_captions` dataset.
     - Generate images with Stable Diffusion.
     - Compute SHAP explanations.
     - Create and display visualizations.
   - Outputs (images, plots) are saved in the working directory.
4. **Hardware Requirements**:
   - GPU with ≥8 GB VRAM (e.g., Colab T4).
   - ~20 GB disk space for models and outputs.
5. **Expected Runtime**:
   - ~2 hours for 10 prompts (generation + SHAP + visualization) on Colab T4.

## References
- Hugging Face Diffusers: [https://huggingface.co/docs/diffusers](https://huggingface.co/docs/diffusers)
- SHAP Documentation: [https://shap.readthedocs.io](https://shap.readthedocs.io)
- Stable Diffusion 2.1: [https://huggingface.co/stabilityai/stable-diffusion-2-1](https://huggingface.co/stabilityai/stable-diffusion-2-1)
- NLPHUJI COCO Captions Dataset: [https://huggingface.co/datasets/nlphuji/coco_captions](https://huggingface.co/datasets/nlphuji/coco_captions)
- Tutorials:
  - “Text-to-Image with Diffusers” (Hugging Face)
  - “SHAP Explained” by StatQuest (YouTube)
  - “Fine-Tuning Stable Diffusion with LoRA” (Hugging Face)
