# Scores

This directory contains similarity matrices between reference `nii` CT scans determined in different ways.

## Human expert scores

These are the scores provided by human experts, which are used as a reference for evaluating the performance of CT embedding models. These scores are characterized by a very low number of labelled images, so results should be interpreted with caution. Furthermore, the scores are not symmetric, meaning that the similarity between two images is not necessarily the same in both directions according to the human experts, *likely an error*, which should be taken into account when interpreting the results.

## Lighter feature extractor scores

These scores are generated using the [Lighter feature extractor](https://huggingface.co/project-lighter/ct_fm_feature_extractor), which is a method for extracting an embedding from CT scans, which are them used to compute cosine similarity matrices.

### Original images

These scores are generated using the original CT scans, without any additional preprocessing.

### Cropped images

CT scans are cropped to the lung region using a lung segmentation model.

TODO! Update and correct description when available.

### Masked images

TODO! Update and correct description when available.

### Masked and resized images

TODO! Update and correct description when available.

## ResNet feature extractor scores

These scores are generated using a ResNet18-based feature extractor. CT scans are preprocessed by segmenting and cropping the lung region, extracting 18 coronal slices, normalizing to 8-bit images, and resizing to 224x224. Features are obtained from a pre-trained ResNet18 (excluding the final layer), concatenated across slices, and cosine similarity matrices are computed between patient embeddings.

## Air Resnet feature extractor scores

Analogous to the ResNet feature extractor scores, but using the air distribution.

TODO! Update and correct description when available.

## Score correlation tests and tnterpretation

The analysis compares human expert similarity judgments of CT scans with similarity estimates derived from several machine learning feature extraction methods. The number of annotated data points is very limited: Mantel tests are based on **6 labeled pairs**, while the other correlation tests use **30 labeled comparisons**. This severely constrains statistical power, and results should therefore be interpreted as preliminary.

To evaluate the correspondence between human expert similarity judgments and machine learning–derived similarity scores, several correlation tests were employed. Pearson and Spearman correlations were computed to assess linear and rank-based associations, respectively, while Kendall’s Tau provided an additional non-parametric rank-based measure. In addition, Mantel tests (Pearson and Spearman variants) were performed to compare similarity matrices directly; however, these were based on only six annotated scan pairs, limiting their reliability.

### Correlation Results

Table 1 summarizes the observed correlations between human expert similarity scores and machine learning feature extractors. Statistically significant correlations (*p* < 0.05) are highlighted in bold.

| Model / Preprocessing          | Metric              | Correlation     | *p*-value     | Samples     |
| ------------------------------ | ------------------- | --------------- | ------------- | ----------- |
| **ResNet18-based air**         | **Mantel Pearson**  | **0.595**       | **0.0069**    | **6**       |
|                                | **Mantel Spearman** | **0.593**       | **0.0139**    | **6**       |
|                                | **Pearson**         | **0.564**       | **0.0012**    | **30**      |
|                                | **Spearman**        | **0.540**       | **0.0021**    | **30**      |
|                                | **Kendall’s Tau**   | **0.444**       | **0.0029**    | **30**      |
| **Lighter (masked + resized)** | **Mantel Pearson**  | **0.529**       | **0.0375**    | **6**       |
|                                | Mantel Spearman     | 0.408           | 0.121         | 6           |
|                                | **Pearson**         | **0.502**       | **0.0047**    | **30**      |
|                                | **Spearman**        | **0.381**       | **0.0375**    | **30**      |
|                                | **Kendall’s Tau**   | **0.302**       | **0.0288**    | **30**      |
| **ResNet18 (baseline)**        | Mantel Pearson      | 0.310           | 0.213         | 6           |
|                                | Mantel Spearman     | 0.318           | 0.235         | 6           |
|                                | Pearson             | 0.294           | 0.114         | 30          |
|                                | Spearman            | 0.287           | 0.124         | 30          |
|                                | Kendall’s Tau       | 0.226           | 0.101         | 30          |
| **Lighter (original)**         | Mantel Pearson      | 0.188           | 0.581         | 6           |
|                                | Mantel Spearman     | 0.322           | 0.236         | 6           |
|                                | Pearson             | 0.178           | 0.346         | 30          |
|                                | Spearman            | 0.261           | 0.164         | 30          |
|                                | Kendall’s Tau       | 0.186           | 0.178         | 30          |
| **Lighter (cropped)**          | Mantel Pearson      | –0.278          | 0.346         | 6           |
|                                | Mantel Spearman     | –0.185          | 0.550         | 6           |
|                                | Pearson             | –0.263          | 0.160         | 30          |
|                                | Spearman            | –0.164          | 0.388         | 30          |
|                                | Kendall’s Tau       | –0.121          | 0.382         | 30          |
| **Lighter (masked only)**      | Mantel Pearson      | –0.356          | 0.174         | 6           |
|                                | Mantel Spearman     | –0.361          | 0.193         | 6           |
|                                | Pearson             | –0.338          | 0.068         | 30          |
|                                | Spearman            | –0.274          | 0.143         | 30          |
|                                | Kendall’s Tau       | –0.226          | 0.101         | 30          |

### Interpretation

The **ResNet18-based air embeddings** achieved the highest and most consistent correlations with expert judgments, with statistically significant results across all metrics. The **Lighter masked + resized variant** also showed moderate and partly significant correlations, indicating some alignment with human perception. In contrast, the **baseline ResNet18**, as well as the **Lighter original, cropped, or masked-only variants**, yielded weak or non-significant results, with some negative correlations observed.

### Conclusion

ResNet18-air embeddings provide the most promising alignment with radiologist similarity assessments, followed by the Lighter masked + resized approach. However, due to the *very limited number of labeled comparisons*, these findings should be regarded as preliminary and require validation on a substantially larger dataset.
