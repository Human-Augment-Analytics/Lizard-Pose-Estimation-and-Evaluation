# Lizard-Pose-Estimation-and-Evaluation

## DLC 
DeepLabCut is an open-source, deep learning-based tool designed for precise, markerless pose estimation of user-defined body parts across various species and behaviors. Developed by Mathis[1] in 2018, it utilizes transfer learning to fine-tune deep neural networks on limited datasets, enabling accurate tracking without the need for physical markers. This approach facilitates the study of complex motor behaviors in naturalistic settings, offering a significant advantage over traditional marker-based systems.

 [1]: https://www.mackenziemathislab.org/deeplabcut

### Configuration set-up:
The training configuration for this DeepLabCut model leverages a ResNet50 backbone with Group Normalization and 2048 output channels, optimized for keypoint detection using Gaussian heatmap generation. The model employs weighted loss functions: MSE for heatmap predictions and Huber for location refinement, ensuring precise detection. Data preprocessing includes RGB normalization and augmentations such as affine transformations (rotation up to ±30° and scaling) and Gaussian noise, while maintaining minimal lateral transformations and no histogram equalization or motion blur. Images are resized dynamically, with scales between 0.4 and 1.0 and a shorter side ranging from 128 to 1152 pixels.
The training process uses an AdamW optimizer with a learning rate of 0.0001 and a two-phase learning rate scheduler to refine performance across 200 epochs. The batch size is set to 1, and snapshots are saved every 25 epochs, with test mAP serving as the key evaluation metric, assessed every 10 epochs. This setup, running on an auto-detected device, is optimized for robust generalization and high accuracy in tracking animal movements, particularly in climbing behavior analysis.

### Data Processing:
The DeepLabCut GUI is used to create labeled data frames. The K-means algorithm was used for to extract X frames for annotation. Each video were extracted 20-30 frames where lizard was moving/still, and 20 vidoes were extracted and labeled in total. Each anole was labeled consistently at the head, upper spine, lower spine, left hindleg, right hindleg and tail. The labeled file was stored in .csv (position of each bodypart points) and .h5 format file and prepared for training step. 
[image_to_add]

### Model visualization:
image to add 

### trajectory plot 

## B-SOiD 

B-SOiD (Behavioral Segmentation of Open-field In Deep Learning) allows users to find behaviors using unsupervised learning, without the need for behavior-annotated data. Specifically, B-SOiD finds clusters in animal behavior using pose estimation data from another tool such as DeepLabCut. B-SOiD begins by extracting pose relationships like distance, speed, and relative angle. Next, it performs a non-linear transformation called UMAP to re-frame data in a lower-dimensional space. Then, HDBSCAN is used to identify clusters, and the clustered features are fed as input to a random forest classifier. In the Python implementation, scikit-learn's RandomForestClassifier is used for this step. Finally, the classifier can used to predict behavior categories in any related data.

![](https://github.com/Human-Augment-Analytics/Lizard-Pose-Estimation-and-Evaluation/blob/main/Behavioral%20Analysis/B-SOiD/Sample%20Gifs/example-side-by-side-shortened.gif)

## VAME

VAME (Video-based Animal Motion Estimation) finds patterns in animal movement with a focus on finding repetitive behaviors. Like B-SOiD, VAME uses pose estimation files from a program like DeepLabCut to identify motifs in animal behavior. VAME first aligns pose estimation data egocentrically and splits the data into fixed-length time windows. Next, it uses a bi-directional recurrent neural network (biRNN) with an encoder-decoder architecture within a Variational Autoencoder (VAE) framework to learn latent representations of the data. After that, both reconstruction and prediction decoders are used to ensure that the latent space captures both reconstruction and prediction. The data is then embedded into the final latent space, and a Hidden Markov Model (HMM) is used to segment the latent space into behavioral motifs.
