# Bangladeshi Shrimp Species Detection Using Deep Learning

A deep learning-powered computer vision system to accurately identify and classify four major Bangladeshi shrimp species using Convolutional Neural Networks (CNN).

## 🦐 Overview

Shrimp, known as "white gold" in Bangladesh, constitutes about 70% of the country's exported agricultural food and represents the second-largest export item. This project addresses the common confusion in identifying different shrimp species by developing an intelligent machine learning system that can accurately recognize shrimp species.

**Research Paper**: [A Deep Learning Approach to Recognize Bangladeshi Shrimp Species](https://www.researchgate.net/publication/375874920_A_Deep_Learning_Approach_to_Recognize_Bangladeshi_Shrimp_Species)

## 🎯 Problem Statement

In Bangladesh, approximately 56 shrimp species are found, but most people, including fishermen, struggle to differentiate between species due to their similar appearances. The most commonly confused pairs are:
- **Golda** and **Bagda** (similar appearance)
- **Horina** and **Deshi** (similar appearance)

This confusion affects:
- Export quality control
- Species-specific pricing
- Market standardization
- Prevention of fraudulent mixing of species

## 🐾 Target Species

The system identifies four major Bangladeshi shrimp species:

1. **Bagda** - High-value export species
2. **Deshi** - Local freshwater variety
3. **Golda** - Premium freshwater prawns
4. **Horina** - Marine shrimp variety

## 🏗️ Project Structure

```
Shrimp-Species-Detection-By-Deep-Learning/
├── OpenCV Implementation/          # Computer vision implementation
├── algorithm_implementation/       # Core CNN models and training scripts
├── data_preprocessing/            # Data preprocessing and augmentation
└── README.md                     # Project documentation
```

## 🧠 Model Architecture

Three custom CNN architectures were developed and compared:

### Model 1 (Complex Architecture)
- **Layers**: 12 convolutional layers
- **Features**: BatchNormalization, mixed pooling (AvgPool2D + MaxPool2D)
- **Accuracy**: 99.01%
- **Epochs**: 44

### Model 2 (Intermediate Architecture)
- **Layers**: 10 convolutional layers
- **Features**: MaxPool2D only
- **Accuracy**: 98.97%
- **Epochs**: 31

### Model 3 (Lightweight Architecture) ⭐ **SELECTED**
- **Layers**: 8 convolutional layers
- **Features**: Alternating Conv2D and MaxPool2D
- **Accuracy**: 99.01%
- **Epochs**: 45
- **Test Accuracy**: 98%

**Why Model 3?** Despite having the same training accuracy as Model 1, Model 3 demonstrated superior performance on the test dataset while being more lightweight and computationally efficient.

## 📊 Dataset

- **Total Images**: 38,042 (after augmentation)
- **Original Images**: 11,500
- **Classes**: 4 (Bagda, Deshi, Golda, Horina)
- **Background**: Black and white backgrounds
- **Data Split**: Training/Validation/Test
- **Test Images**: 3,122

### Data Augmentation Techniques
- Rotation
- Angle adjustment
- Various transformations to increase dataset diversity

## 🎯 Performance Metrics

### Model 3 (Final Model) Performance:

| Species | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| Bagda   | 0.97      | 1.00   | 0.98     | 569     |
| Deshi   | 1.00      | 0.96   | 0.98     | 874     |
| Golda   | 0.98      | 0.98   | 0.98     | 1067    |
| Horina  | 0.97      | 0.99   | 0.98     | 612     |

**Overall Accuracy**: 98%

## 🛠️ Technical Stack

- **Deep Learning Framework**: TensorFlow, Keras
- **Computer Vision**: OpenCV
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn
- **Programming Language**: Python
- **Hardware**: NVIDIA GTX 1650 (training)

## 🖥️ Implementation

The system includes:
- **Real-time Detection**: OpenCV implementation for live species recognition
- **Contour Detection**: Automatic shrimp boundary detection
- **Species Classification**: Real-time species identification with confidence scores
- **Bounding Box**: Visual identification with species labeling

### Color Range for Mask Generation:
- **Lower Range**: (30, 30, 30)
- **Higher Range**: (170, 200, 200)

## 📸 Detection Results

The OpenCV implementation successfully detects and classifies shrimp species with visual bounding boxes and labels:

| Species | Input Image | Detected Output |
|---------|-------------|-----------------|
| **Bagda** | <img src="OpenCV Implementation/bagda.jpg" width="300", height="300"> | <img src="OpenCV Implementation/Detected Bagda final.jpg" width="300", height="300"> |
| **Deshi** | <img src="OpenCV Implementation/deshi.jpg" width="300", height="300"> | <img src="OpenCV Implementation/Detected Deshi final.jpg" width="300, height="300""> |
| **Golda** | <img src="OpenCV Implementation/golda.jpg" width="300", height="300"> | <img src="OpenCV Implementation/Detected Golda final.jpg" width="300", height="300"> |
| **Horina** | <img src="OpenCV Implementation/horina.jpg" width="300", height="300"> | <img src="OpenCV Implementation/Detected Horina final.jpg" width="300", height="300"> |

*The detected images show the system's ability to accurately identify species with bounding boxes and confidence labels.*

## 🚀 Getting Started

### Prerequisites
```bash
pip install tensorflow
pip install opencv-python
pip install scikit-learn
pip install pandas
pip install numpy
```

### Usage
1. Clone the repository
2. Navigate to the appropriate directory:
   - `algorithm_implementation/` for model training
   - `OpenCV Implementation/` for real-time detection
   - `data_preprocessing/` for data preparation
3. Run the desired scripts

## 📈 Results

- **Training Accuracy**: 99.01%
- **Test Accuracy**: 98%
- **No Overfitting**: Validation loss converges with training loss
- **Species Recognition**: Highly accurate identification across all four species

### Prediction Results:
- **Bagda**: 569/569 images correctly predicted (100%)
- **Deshi**: 839/874 images correctly predicted (96%)
- **Golda**: 1046/1067 images correctly predicted (98%)
- **Horina**: 606/612 images correctly predicted (99%)

## 🎯 Applications

1. **Export Quality Control**: Ensure correct species labeling for international markets
2. **Fraud Prevention**: Detect mixing of different species for profit
3. **Educational Tool**: Help fishermen and buyers identify species correctly
4. **Market Standardization**: Promote consistent species classification
5. **Mobile Application**: Future development for on-field identification

## 🔮 Future Enhancements

- [ ] Expand to all 56 Bangladeshi shrimp species
- [ ] Increase dataset size without augmentation
- [ ] Develop mobile applications (Android & iOS)
- [ ] Implement deeper CNN architectures with better computational resources
- [ ] Real-time batch processing for commercial applications
- [ ] Integration with IoT devices for automated sorting

## 📚 Research Impact

This project contributes to:
- **Agricultural Technology**: AI integration in aquaculture
- **Export Industry**: Quality assurance for Bangladesh's second-largest export
- **Computer Vision**: Species classification in challenging aquatic environments
- **Deep Learning**: Lightweight model development for practical applications

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- Model improvements
- Dataset expansion
- New feature implementations
- Bug fixes

## 📄 License

This project is available for academic and research purposes. Please cite the original research paper when using this work.

## 📞 Contact

For questions, collaborations, or research inquiries, please refer to the research paper or open an issue in this repository.

---

**Citation**: If you use this work in your research, please cite:
```
Hasan, M. M., Nishi, J. S., Habib, M. T., Islam, M. M., & Ahmed, F. (2023). 
A Deep Learning Approach to Recognize Bangladeshi Shrimp Species. 
14th ICCCNT IEEE Conference, IIT-Delhi, Delhi, India.
```

## 🏆 Achievements

- **99.01% Training Accuracy** achieved with lightweight architecture
- **98% Test Accuracy** with robust generalization
- **Real-time Implementation** using OpenCV
- **Practical Application** for Bangladesh's shrimp export industry

---

*This project demonstrates the successful application of deep learning in solving real-world agricultural challenges in Bangladesh's aquaculture sector.*