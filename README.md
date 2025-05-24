# Vision Transformer (ViT) Implementation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1EVShP_ej_pJ0T9pyo3juZl8nZwL32Hty?usp=sharing)

PyTorch implementation of ["An Image is Worth 16x16 Words"](https://arxiv.org/abs/2010.11929) in a single Google Colab notebook.

![Colab Screenshot](./assets/colab_screenshot.png)  
*Interactive Vision Transformer implementation with demo capabilities*

## Notebook Features
🧠 **Paper-Focused Implementation**
- Complete ViT-B/16 architecture (12 layers, 768D embeddings)
- Step-by-step code matching paper sections
- Hyperparameters from Table 1

🔍 **Interactive Components**
- Live demo with image upload
- Attention pattern visualization
- Patch decomposition viewer

📈 **Training & Evaluation**
- Food-101 subset integration (pizza/steak/sushi)
- Training progress tracking
- Accuracy/loss metrics

## Get Started
1. Click the **Open in Colab** button above
2. In Colab:  
   a. Select `Runtime > Run all`  
   b. Use the demo cell to upload food images  
   c. Modify hyperparameters in the config section

## Key Components
| Section | Description | Paper Reference |
|---------|-------------|-----------------|
| `Patch Embeddings` | 16x16 image splitting | Eq. 1 |
| `Positional Encoding` | Learnable 1D positions | Sec 3.1 |
| `Transformer Encoder` | 12-layer attention stack | Fig 1 |
| `[CLS] Token` | Classification aggregation | Sec 3.1 |

## Results Preview
```python
Epoch 10/50 | Train Loss: 0.21 | Val Acc: 89.3%
