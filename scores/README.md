# Scores

This directory contains similarity matrices between reference `nii` CT scans determined in different ways.

## Human expert scores

These are the scores provided by human experts, which are used as a reference for evaluating the performance of CT embedding models. These scores are characterized by a very low number of labelled images, so results should be interpreted with caution. Furthermore, the scores are not symmetric, meaning that the similarity between two images is not necessarily the same in both directions according to the human experts, *likely an error*, which should be taken into account when interpreting the results.

## Lighter feature extractor scores

These scores are generated using the [Lighter feature extractor](https://huggingface.co/project-lighter/ct_fm_feature_extractor), which is a method for extracting an embedding from CT scans, which are them used to compute cosine similarity matrices.

## ResNet feature extractor scores

These scores are generated using a ResNet18-based feature extractor. CT scans are preprocessed by segmenting and cropping the lung region, extracting 18 coronal slices, normalizing to 8-bit images, and resizing to 224x224. Features are obtained from a pre-trained ResNet18 (excluding the final layer), concatenated across slices, and cosine similarity matrices are computed between patient embeddings.

## Air Resnet feature extractor scores

TODO!
