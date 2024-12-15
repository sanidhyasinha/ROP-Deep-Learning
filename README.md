# **Retinopathy of Prematurity (ROP) Classification Using Fundus Images and Deep Learning**

## **Overview**
This project focuses on the **detection and classification of Retinopathy of Prematurity (ROP)** stages from retinal fundus images using advanced **Deep Learning** techniques, particularly **Convolutional Neural Networks (CNNs)** and **Transfer Learning**. ROP is a major cause of preventable blindness in premature infants, and this automated system aims to assist ophthalmologists in early and accurate diagnosis.

By leveraging deep learning algorithms, the project provides:
- Detection of abnormal blood vessel growth and tortuosity.
- Multi-class classification of ROP stages (Stage 1 to Stage 5).
- Identification of **plus disease** associated with severe ROP.

---

## **Table of Contents**
1. [Introduction](#introduction)  
2. [Dataset](#dataset)  
3. [Preprocessing Steps](#preprocessing-steps)  
4. [Model Architecture](#model-architecture)  
5. [Installation](#installation)  
6. [Usage](#usage)  
7. [Project Workflow](#project-workflow)  
8. [Results](#results)  
9. [Evaluation Metrics](#evaluation-metrics)  
10. [Visualizations](#visualizations)  
11. [Challenges](#challenges)  
12. [Future Scope](#future-scope)  
13. [Contributing](#contributing)  
14. [License](#license)  
15. [Acknowledgements](#acknowledgements)  

---

## **Introduction**
**What is Retinopathy of Prematurity?**
-ROP is an eye disease affecting premature infants with low birth weight (<3 pounds).
-It occurs due to abnormal blood vessel growth in the retina.

 **Why is ROP Important?**
-A leading cause of visual impairment and blindness in premature infants.
-Early detection can prevent severe vision loss.

**What is Fundus Image?**
Medical images of the retina showing blood vessels and optic structures.

**Stages Of Retinopathy of Prematurity:**
-Stage 1 ROP: Demarcation of Line                 
-Stage 2 ROP: Visible ridge
-Stage 3 ROP: Blood vessels in the ridge         
-Stage 4 ROP: Sub-total retinal detachment
-Stage 5 ROP: Total retinal detachment       

![Screenshot 2024-12-11 130105](https://github.com/user-attachments/assets/49299e3d-7ef2-4ead-8082-b40caa3d60cd)


---

## **Dataset**
The model is trained and tested on publicly available **fundus image datasets** for ROP classification.  

### **Dataset Details**:
- **Source**: Kaggle ROP dataset, DRIVE dataset, or custom clinical data.
- **Labels**:  
  - Normal  
  - Stage 1 (Mild ROP)  
  - Stage 2 (Moderate ROP)  
  - Stage 3 (Severe ROP with ridge formation)  
  - Stage 4 (Partial retinal detachment)  
  - Stage 5 (Complete retinal detachment)  
  - Plus Disease
 
  - ![Screenshot 2024-09-26 193359](https://github.com/user-attachments/assets/cbebbd01-fa80-4e2e-8953-fdf5fae391c6)


### **Folder Structure**:
```bash
data/
    ├── train/         # Training images
    ├── test/          # Test images
    ├── val/           # Validation images
    ├── annotations/   # Image labels
```

---

## **Preprocessing Steps**
Fundus images are preprocessed for optimal input to the deep learning model:
1. **Image Resizing**: Standardized to 224x224 pixels.
2. **Normalization**: Scale pixel values between 0–1.
3. **Noise Removal**: Gaussian filters to reduce image noise.
4. **Contrast Enhancement**: Histogram equalization or CLAHE.
5. **Data Augmentation**:
   - Random rotation
   - Horizontal flipping
   - Brightness/contrast adjustments
     
![CNN](https://github.com/user-attachments/assets/0a1f2578-0636-4e27-b277-8c9ea0abda1f)

---

## **Model Architecture**

### **1. Convolutional Neural Networks (CNNs)**
The project uses CNN-based models for image classification:
- **Custom CNN**:  
  Consists of convolutional layers, max-pooling layers, and fully connected layers.

### **2. Transfer Learning**
Pre-trained models are fine-tuned on the ROP dataset:
- **CNN-19**

![Screenshot 2024-12-10 225836](https://github.com/user-attachments/assets/ade1db59-2dec-4f08-b0b1-90051f2b0193)

![Screenshot 2024-12-10 111304](https://github.com/user-attachments/assets/9dbfca6d-663b-420a-83f0-1b1a21be5583)



---

## **Installation**

Follow these steps to set up the project locally.

### **1. Clone the Repository**
```bash
git clone https://github.com/your-username/rop-classification.git
cd rop-classification
```

### **2. Create a Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### **3. Install Dependencies**
Install the required libraries using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### **4. Prepare the Dataset**
- Download the dataset and organize it in the `data/` directory.

---

## **Usage**

### **1. Train the Model**
Train the ROP classification model with the following command:
```bash
python train.py --dataset data/ --epochs 50 --model resnet50 --batch_size 16
```

### **2. Test the Model**
Evaluate the trained model on the test set:
```bash
python test.py --weights saved_model.pth --dataset data/test/
```

### **3. Visualize Predictions**
Visualize the critical regions (e.g., vessel abnormalities) using Grad-CAM:
```bash
python visualize.py --image data/sample.jpg --weights saved_model.pth
```

---

## **Project Workflow**

1. **Data Collection**: Fundus images labeled with ROP stages.
2. **Preprocessing**: Enhance image quality and augment data.
3. **Model Training**: Train CNN models with transfer learning.
4. **Evaluation**: Evaluate accuracy, AUC-ROC, and other metrics.
5. **Visualization**: Generate explainable results using Grad-CAM.

---

## **Results**

### **Model Performance**:

![Screenshot 2024-12-11 215221](https://github.com/user-attachments/assets/61a802cb-f7d0-4b80-bc5c-2125ae8b346e)

### **Example Outputs**:
- Correct classification of ROP stages.
- Visualization of regions affected by plus disease.

---

## **Evaluation Metrics**
- **Accuracy**
- **Precision, Recall, and F1-Score**
- **AUC-ROC** (Area Under the Curve)
- **Confusion Matrix**

---

## **Visualizations**
- **Grad-CAM**: Highlights areas of interest in fundus images (e.g., blood vessels).
- **Loss and Accuracy Plots**: Training and validation performance graphs.

---

## **Challenges**
- **Limited Data**: Scarcity of publicly available labeled ROP datasets.
- **Class Imbalance**: Fewer severe-stage ROP samples.
- **Variability**: Differences in image quality and annotations.

---

## **Future Scope**
- Integration with **Edge AI** devices for real-time screening.
- Multi-modal analysis combining fundus images with clinical data.
- Deployment as a **telemedicine tool** for rural areas.

---

## **Contributing**
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit and push changes:
   ```bash
   git commit -m "Add feature description"
   git push origin feature-name
   ```
4. Submit a pull request.

---

## **License**
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## **Acknowledgements**
- **Open-Source Datasets**: DRIVE, Kaggle, and other ROP datasets.
- **Libraries**: PyTorch, TensorFlow, OpenCV, NumPy, and Matplotlib.
- Thanks to medical experts for labeling data and validating the results.
