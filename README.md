
# 📌 Minist-Digits-Recognizer-in-C

![MNIST Sample](res/digits.png)

---

## 🧠 Introduction

The **MNIST handwritten digits dataset** is one of the most widely used benchmark datasets in machine learning.

This project implements a **Neural Network from scratch in pure C** to classify handwritten digits (0–9).

- 📷 Image size: 28 × 28 (grayscale)
- 🔢 Number of samples: 42,031
- 🏋 Training set: 33,625 images
- 🧪 Test set: 8,406 images
- 🎯 Classes: 10 (digits 0–9)

---

## 🎯 Objective

- Build a neural network **without external ML libraries**
- Implement:
  - Matrix operations
  - Forward propagation
  - Backpropagation
  - Softmax activation
  - Cross-entropy loss
  - Train/Test split
- Achieve high accuracy on unseen data

---

## 🏗 Architecture

Current model:

```

Input Layer  : 784 neurons (28x28 flattened)
Hidden Layer : 512 neurons (ReLU)
Output Layer : 10 neurons (Softmax)

```

---

## 📊 Results

```

Train Accuracy: 96%
Test Accuracy:  94%
Test Loss:      0.0127

````

The model generalizes well with minimal overfitting.

---

## ⚙️ Build & Run

### 🔧 Build (Linux / macOS)

```bash
make
````

Or Download [Here](https://github.com/lotfiab1/Minist-Digits-Recognizer-in-C/blob/master/build.sh) :

Then run :

```bash
chmod +x build.sh
```
```bash
./build.sh
```

### 🔧 Build (Windows)

```bash
make
````
Or Download [build.bat](https://github.com/lotfiab1/Minist-Digits-Recognizer-in-C/blob/master/build.bat)

```bash
./build.bat
```

### ▶ Run

```bash
./mnist
```

or (Windows):

```bash
mnist.exe
```

---

## 📂 Project Structure

```
Minist-Digits-Recognizer-in-C/
│
├── include/
├── lib/
|    ├── libopenblas.dll
|
├── modeles/
├── res/
|   ├── dataset.csv
|
├── src/
│   ├── dataset.c
│   ├── datatypes.c
│   ├── fileshandler.c
│   ├── llm-digits-minist-model.c
|   ├── main.c
|   ├── math-numpy.c
|   ├── resourceshandler.c
│
|
├── Makefile
├── build.sh
├── build.bat
└── README.md
```

---

## 🚀 Future Improvements

* [ ] Add Convolutional Neural Network (CNN)
* [ ] Mini-batch gradient descent
* [ ] SIMD optimization
* [ ] Confusion matrix
* [ ] Performance benchmarking

---

## 🛠 Technologies

* Language: **C**
* No external ML libraries
* Manual memory management
* Fully custom backpropagation implementation

---

## 📜 License

MIT License
