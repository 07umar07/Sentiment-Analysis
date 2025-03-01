# Sentiment Analysis with RNN

## Overview
This project implements a sentiment analysis model using a **Recurrent Neural Network (RNN)** with **Bidirectional LSTM**. The model is trained to classify text into **seven different sentiment categories**.

## Dataset
The dataset used in this project consists of various text samples labeled with sentiments ranging from 0 to 6. The dataset is split as follows:
- **Train:** 99%
- **Cross-Validation (CV):** 0.5%
- **Test:** 0.5%

## Data Preparation
1. **Text Processing:** Tokenization and text embedding.
2. **Label Mapping:** Mapping sentiment labels to integer values (0-6).
3. **Data Splitting:** Splitting the dataset into training, validation, and testing sets.

## Model Architecture
The model consists of:
- **Embedding Layer**: 100 x 128 input dimension with regularization.
- **Bidirectional LSTM Layers**: Three LSTM layers, with the first two having `return_sequences=True` to use outputs from all timesteps.
- **Fully Connected Layer**: 7 output units with **softmax activation**.

The model is compiled using:
- **Loss Function:** Sparse Categorical Crossentropy
- **Optimizer:** Adam
- **Metrics:** Accuracy

## Training
- **Epochs:** 10
- **Batch Size:** 16
- **Validation Set:** `X_cv`, `y_cv`
- **Cross-validation** is used to enhance the model's performance with unseen data during training.

## Model Evaluation
The trained model is evaluated using:
- **Test Set Accuracy**
- **F1 Score** for performance measurement

## Saving the Model
The trained model and its weights are saved for future inference.

## Dependencies
- TensorFlow
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## License
This project is licensed under **GPL-3.0**. See the [LICENSE](LICENSE) file for details.

