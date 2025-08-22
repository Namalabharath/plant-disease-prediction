# 🌱 Plant Disease Detection & Treatment Assistant

An intelligent AI-powered web application for detecting plant diseases and providing comprehensive treatment recommendations using Convolutional Neural Networks (CNN) and Streamlit.

## 🎯 Features

- **AI-Powered Disease Detection**: CNN model trained on 50,000+ images with 95% accuracy
- **38 Disease Classes**: Covers 14 plant species including Apple, Tomato, Corn, Grape, Potato, and more
- **Interactive Web Interface**: Modern Streamlit UI with image enhancement controls
- **Comprehensive Treatment Database**: 200+ treatment recommendations with medicines, fertilizers, and prevention methods
- **Real-time Image Processing**: Upload, enhance, and analyze plant images instantly
- **Downloadable Reports**: Get treatment recommendations in JSON format

## 🚀 Demo

### Disease Detection Interface
- Upload plant leaf images (JPG, PNG)
- Real-time image enhancement (brightness, contrast, saturation)
- Instant disease prediction with treatment recommendations

### Treatment Recommendations
- **💊 Medicines**: Specific fungicides, bactericides, and treatments
- **🌿 Fertilizers**: NPK ratios and organic amendments  
- **🛡️ Prevention**: Best practices and cultural controls
- **🌱 Growing Conditions**: Optimal soil types and temperature ranges

## 🛠️ Technology Stack

- **Machine Learning**: TensorFlow 2.15, Keras, CNN Architecture
- **Web Framework**: Streamlit with custom CSS styling
- **Data Processing**: NumPy, Pandas, PIL (Python Imaging Library)
- **Visualization**: Plotly for interactive charts
- **Model**: Custom CNN with 2 convolutional layers, MaxPooling, and dense layers

## 🏗️ Architecture

```
Input Image (224x224) → CNN Model → Softmax → Disease Prediction → Treatment Database → Recommendations
```

### Model Details
- **Input Size**: 224x224x3 (RGB images)
- **Architecture**: Sequential CNN with MaxPooling
- **Classes**: 38 plant disease categories
- **Accuracy**: ~95% on validation set
- **Training Data**: PlantVillage dataset (50,000+ images)

## 📁 Project Structure

```
plant-disease-detection/
├── app/
│   ├── main.py              # Streamlit web application
│   ├── requirements.txt     # Python dependencies
│   ├── class_indices.json   # Disease class mappings
│   ├── remedies.json       # Treatment database
│   └── trained_model/      # CNN model files (not included - too large)
├── model_training_notebook/
│   └── Plant_Disease_Prediction_CNN_Image_Classifier.ipynb
├── test_images/            # Sample test images
└── README.md
```

## 🚦 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/YOUR-USERNAME/plant-disease-detection.git
cd plant-disease-detection
```

2. **Install dependencies**
```bash
cd app
pip install -r requirements.txt
```

3. **Download the trained model**
> **Note**: The trained model file (547MB) is not included in the repository due to GitHub's file size limits.

**Option 1**: Train your own model using the provided notebook
```bash
# Open and run the Jupyter notebook
jupyter notebook model_training_notebook/Plant_Disease_Prediction_CNN_Image_Classifier.ipynb
```

**Option 2**: Download pre-trained model
- Contact the repository owner for the pre-trained model file
- Place `plant_disease_prediction_model.h5` in `app/trained_model/`

4. **Run the application**
```bash
streamlit run main.py
```

5. **Open your browser**
Navigate to `http://localhost:8501`

## 🌿 Supported Plant Species & Diseases

### Plants Covered (14 species):
- 🍎 **Apple**: Apple Scab, Black Rot, Cedar Apple Rust
- 🍇 **Grape**: Black Rot, Esca, Leaf Blight  
- 🍅 **Tomato**: Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus
- 🥔 **Potato**: Early Blight, Late Blight
- 🌽 **Corn**: Cercospora Leaf Spot, Common Rust, Northern Leaf Blight
- 🍑 **Cherry**: Powdery Mildew
- 🍊 **Orange**: Huanglongbing (Citrus Greening)
- 🍑 **Peach**: Bacterial Spot
- 🌶️ **Pepper**: Bacterial Spot
- 🫐 **Blueberry**, 🍓 **Strawberry**, 🥒 **Squash**, 🫛 **Soybean**, 🍇 **Raspberry**

## 📊 Model Performance

- **Training Accuracy**: ~95%
- **Validation Split**: 80/20
- **Image Augmentation**: Rescaling, rotation, zoom
- **Batch Size**: 32
- **Epochs**: 5 (adjustable)
- **Optimizer**: Adam

## 🎨 UI Features

### Modern Interface
- **Responsive Design**: Works on desktop and mobile
- **Custom Styling**: Gradient backgrounds and modern cards
- **Image Enhancement**: Real-time brightness, contrast, saturation controls
- **Tabbed Layout**: Organized treatment information

### Interactive Elements
- **File Upload**: Drag-and-drop image upload
- **Image Preview**: Enhanced image display
- **Treatment Tabs**: Medicines, Prevention, Growing Conditions, Summary
- **Download Reports**: JSON format treatment reports

## 🔬 How It Works

1. **Image Upload**: User uploads a plant leaf image
2. **Preprocessing**: Image is resized to 224x224 and normalized
3. **CNN Prediction**: Model processes the image through convolutional layers
4. **Classification**: Softmax layer outputs probability distribution
5. **Treatment Lookup**: System matches prediction to remedies database
6. **Results Display**: Shows disease name and comprehensive treatment plan

## 📈 Future Enhancements

- [ ] **Mobile App**: React Native or Flutter implementation
- [ ] **Real-time Camera**: Direct camera capture functionality
- [ ] **Severity Assessment**: Disease severity scoring
- [ ] **Geographic Recommendations**: Location-based treatment advice
- [ ] **Multi-language Support**: Localization for different regions
- [ ] **Offline Mode**: Local model deployment
- [ ] **API Integration**: RESTful API for third-party integration

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PlantVillage Dataset**: Training data source
- **TensorFlow Team**: Deep learning framework
- **Streamlit**: Web application framework
- **Agricultural Experts**: Domain knowledge for treatment recommendations


⭐ **Star this repository if you find it helpful!**

## 🚨 Important Note

**Model File Not Included**: The trained model file (`plant_disease_prediction_model.h5`) is 547MB and exceeds GitHub's file size limits. Please follow the installation instructions to obtain the model file.

**Disclaimer**: This tool is for educational and research purposes. Always consult with agricultural professionals for critical plant health decisions.
