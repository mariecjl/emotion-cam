![Image](https://github.com/user-attachments/assets/5dd02c94-52f7-4f27-a8d2-a12383caedde)

![Image](https://github.com/user-attachments/assets/08d7af1e-25fe-406d-9f46-d7f7c591e75b)
_*Stock videos above from Shutterstock & Storyblocks_


This projects includes implementation of both a CNN and a combination of MLP + MediaPipe Facemesh trained on the FER2013 dataset to recognize human facial emotions in real time.
The trained model takes frames of a camera feed as input and displays confidence scores as live probability bars. (Test accuracy ~65%)

Architecture
- CNN
  - 3 convolutional blocks (Conv2D -> batchnorm -> ReLU -> maxpool)
  - fully connected layers (256 hidden units & drop out for regularization
  - cross-entropy loss
  - optimizer: adam
- Alternative: MLP + MediaPipe Facemesh
  - Images fed into MediaPipe FaceMesh -> produces 478 facial landmarks (x, y, z)
  - geometric normalization and flattening the landmark vectors
  - MLP for emotion class prediction
    - two blocks: linear -> ReLU -> dropout (0.3)
 


Emotions supported are the labels of the original FER2013 dataset
- Angry, disgust, fear, happy, sad, surprise, neutral
