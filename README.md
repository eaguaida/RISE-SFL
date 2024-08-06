# Causal-RISE
Causal Explanations using Statistical Fault Localisation

This is a hybrid XAI framework taking inspiration from RISE https://arxiv.org/abs/1806.07421 in the creation of Binary Mask as a method of Perturbating the Input of Images, and using Statistical Fault Localisation https://arxiv.org/pdf/1908.02374 as a pixel relevance metric - The output is a Saliency Map with the most relevant pixels for a classification. This framework serves as a localised technique, capable of doing a deep explanation in Image Classifiers on 1 class. It's slower than other XAI techniques counterparts, but achieves better results with a smaller footprint in the GPU.

<p align="center">
  <img src="https://raw.githubusercontent.com/shap/shap/master/docs/artwork/shap_header.svg" width="800" />
</p>

In order to measure my implementation, I used the Causal Metrics introduced in RISE and an implementation of Tristan saliency maps metric in


# How to use it?
### To explain a single image:

from masker.generation import SFL
### To explain a batch of images:

# How does it work?
## Masking Process

## Mutant Generation

## Computing Ranking Procedure

### Set of Parameters
### Fault Localisation 
