# EFM32-Based-ML-Model-Optimization-Deployment


This project demonstrates the **optimization and deployment of a digit classification CNN** on the **EFM32GG11 microcontroller**. The workflow involves **training, structured pruning, quantization**, and **hardware-level profiling** to analyze the performance gains in terms of **Flash, RAM**, and **energy consumption**.

---

## 📁 Project Structure

```
├── ECPS_202_EdgeAI_Report.pdf         # Detailed technical report
├── Part1_Training/                    # CNN training and evaluation
├── Part2_Pruning/                     # L1-norm based structured pruning
├── Part3_Quantization/                # 8-bit post-training quantization
├── Deployment/                        # Final embedded deployment (EFM32GG11)
├── Data/                              # Digit dataset (0-9)
└── README.md                          # You're here!
```

---

## 🧪 Overview

### ✅ Task:
Develop and optimize a CNN for digit classification using MATLAB. Post-training, the model undergoes:

- **Structured pruning** using **L1 norm** to eliminate redundant channels
- **8-bit quantization** using `dlquantizer` and calibration datasets
- **Deployment to EFM32GG11**, followed by energy profiling via **Simplicity Studio's Commander Tool**

### 📌 Tools & Frameworks Used:
- **MATLAB + Deep Learning Toolbox**
- **Embedded C SDK via Simplicity Studio**
- **EFM32GG11 Giant Gecko MCU**
- **Commander Tool** for energy + Flash/RAM profiling

---

## 🔧 Functions and Their Role

### 1. **Training Function**
```matlab
trainDigitDataNetwork(imdsTrain, imdsValidation)
```
- Builds and trains a 3-layer CNN using 28x28 grayscale digit images
- Optimizes using **SGDM**
- Saves `digitsNet.mat` for future use

---

### 2. **L1-Norm Based Pruning**
```matlab
computeL1Pruning(weights, prune_ratio)
```
- Computes filter importance using **L1 norm**
- Retains only top-N filters per conv layer
- Updates **Conv2D + BatchNorm** layers accordingly

```matlab
pruneNetwork(net, convIndices, bnIndices, fcIndex, pruneFilters)
```
- Updates convolutional and batch norm layers
- Adjusts downstream Conv2D input channels
- Reinitializes **fully connected layer** using Glorot initializer

---

### 3. **Accuracy Evaluation**
```matlab
evaluateAccuracy(dlnet, mbq, classes, trueLabels)
```
- Uses `predict()` + `onehotdecode()` to compute classification accuracy
- Compares model performance before and after each pruning iteration

---

### 4. **Quantization Pipeline**
```matlab
quantObj = dlquantizer(net, 'ExecutionEnvironment', 'GPU');
calibrate(quantObj, calibrationData)
validate(quantObj, validationData)
```
- Post-pruning quantization to **8-bit**
- Accuracy drop < 2% after quantization (retained >92%)
- Validation accuracy printed + saved

---

## 📉 Optimization Progress

| Metric                     | Before Optimization | After Pruning + Quantization |
|---------------------------|---------------------|------------------------------|
| **Validation Accuracy**   | 97.5%               | **92.1%**                    |
| **Model Size**            | 27.5 KB             | **18.3 KB**                  |
| **Flash Usage**           | 23.4 KB             | **16.2 KB**                  |
| **RAM Usage**             | 7.2 KB              | **4.9 KB**                   |
| **Energy (avg µJ/image)** | 7.43 µJ             | **4.91 µJ**                  |
| **Sparsity Achieved**     | 90%                 | ✅                           |

---

## 📊 Graphs (to insert later)

- **Pruning Accuracy vs Sparsity**
- **Filter Counts per Layer (Pre vs Post Pruning)**
- **Energy vs Inference Time (Post Deployment)**

_Use screenshots from MATLAB and Simplicity Studio here._

---

## ⚡ Profiling (EFM32GG11)

- Used **Simplicity Studio Commander Tool** to analyze:
  - Flash/RAM before and after optimization
  - Energy consumption using EFM Energy Profiler
- **Significant reduction in RAM and Flash (~30%)**
- **Energy savings >35% per inference**

---

## 🧠 Key Learnings

- **Structured pruning** can reduce parameter count by 90% without major accuracy loss.
- **Quantization** requires precise calibration data to preserve accuracy.
- **Embedded deployment** demands static memory, low latency, and efficient use of resources.
- **Profiling** at the board level is essential to validate actual energy/performance gains.

---

## 📌 Summary

> Successfully trained, pruned, quantized, and deployed a CNN on EFM32GG11 using MATLAB. Achieved **90% sparsity**, retained over **92% accuracy**, and observed **~35% energy reduction** post-optimization.

---

## 📷 Screenshots (to insert)
- MATLAB accuracy plots
- Pruning bar graphs
- Simplicity Studio power profile output
- Flash and RAM usage breakdown

---

## 📁 References

- [MathWorks Structured Pruning Docs](https://www.mathworks.com/help/deeplearning/ref/prune.html)
- [EFM32GG11 Datasheet](https://www.silabs.com/documents/public/data-sheets/efm32gg11-datasheet.pdf)
- [Simplicity Studio Tools](https://www.silabs.com/developers/simplicity-studio)

---

> _Made as part of ECPS 202: Cyber-Physical Systems Design_
