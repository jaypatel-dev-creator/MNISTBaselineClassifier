# MNIST Digit Classification using PyTorch

This project implements a **fully professional training pipeline** for handwritten digit classification on the **MNIST dataset** using **PyTorch**. It is designed as a clean, interview-ready baseline that demonstrates real AI/ML practices such as train/validation/test splits, proper evaluation, and device-aware training.

---

##  Project Overview

- **Task**: Multiclass classification (digits 0–9)
- **Dataset**: MNIST (60,000 train + 10,000 test images)
- **Model Type**: Fully Connected Neural Network (baseline)
- **Framework**: PyTorch

This project intentionally uses a **simple feed-forward network** to focus on **correct ML workflow**, not architectural complexity.

---

##  Key ML Practices Followed

- Train / Validation / Test split (no data leakage)
- GPU/CPU device handling
- Proper use of `model.train()` and `model.eval()`
- Loss **and** accuracy tracking
- Normalized inputs
- Clean inference logic
- Model checkpoint saving

This reflects **real-world ML training discipline**, not just tutorial code.

---




## 🗂️ Dataset Preparation

- MNIST loaded using `torchvision.datasets.MNIST`

### Input Transformations
- `ToTensor()`
- `Normalize(mean=0.1307, std=0.3081)`

### Data Split
- **Training**: 85% of training set
- **Validation**: 15% of training set
- **Test**: Official MNIST test set


---

##  Training Details

- **Loss Function**: `CrossEntropyLoss`
- **Optimizer**: Adam (`lr = 0.001`)
- **Batch Size**: 64
- **Epochs**: 5

During training:
- Average loss per epoch is reported
- Accuracy is computed explicitly

---

##  Evaluation

Metrics reported:
- Training Loss & Accuracy
- Validation Loss & Accuracy
- Final Test Loss & Accuracy

Test evaluation is performed **only once after training**, ensuring unbiased results.

---

##  Inference

The project includes a clean inference example:
- Switches model to evaluation mode
- Disables gradient computation
- Applies softmax only for probability interpretation
- Uses argmax for predicted digit

---

## 💾Model Saving

The trained model is saved using:


torch.save(model.state_dict(), "mnist_model.pth")
This follows PyTorch best practices and allows reuse for inference or further training.

---

## ▶️ How to Run

### 1. Install Dependencies

pip install torch torchvision

### 2. Run Training

Execute the Python script or notebook containing the code.  
The dataset will download automatically.

---

##  Future Improvements

- Replace fully connected model with CNN
- Add learning rate scheduler
- Visualize predictions
- Extend to CIFAR-10
- Add TensorBoard logging

---

##  Learning Outcome

This project demonstrates:
- Correct ML pipeline design
- Proper PyTorch training structure
- Interview-ready code organization

It serves as a **strong foundational project** before moving to CNNs and larger datasets.

---


