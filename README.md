# Smart Bin – Waste Classification Using Deep Learning

This project is an AI-powered Smart Recycling Bin that classifies waste into five categories:
- Plastic  
- Paper  
- Glass  
- Metal  
- General Waste  

It uses TensorFlow/Keras with transfer learning (MobileNetV2) and includes a prediction interface for testing individual images.

---

##  Features
- Transfer Learning using MobileNetV2  
- Model training script  
- Real-time classification interface (`smart_bin_app.py`)  
- Class map generation  
- Training history visualization  
- Pretrained model files included  

---

##  Model Files
The repository includes two trained models:

- `models/smart_bin_mbv2.h5`  
- `models/smart_bin_mbv2_best.keras`  

You can load them in the app with: 

```python
from tensorflow.keras.models import load_model
model = load_model("models/smart_bin_mbv2_best.keras")

```

## Project Structure
```
smart-bin/
│
├── train_mobilenet.py
├── smart_bin_v2.py
├── smart_bin_app.py
├── make_classmap.py
├── plot_history.py
│
├── class_map.json
├── history.json
│
├── models/
│   ├── smart_bin_mbv2.h5
│   └── smart_bin_mbv2_best.keras
│
└── results_img/
    ├── accuracy_curve.png
    ├── loss_curve.png
    └── confusion_matrix.png
```

## How to Run the Project

  ### Install Dependencies
```
pip install tensorflow numpy matplotlib scikit-learn pillow
```

  ### Train the Model
```
python train_mobilenet.py
```

  ### Generate Class Map (Optional)
```
python make_classmap.py
```

  ### Launch the Prediction Interface
```
python smart_bin_app.py
```


## Dataset
The dataset is not included in this repository due to size limitations.

## Results
Training curves and confusion matrix are available in the results_img/ folder.

## Authors
**Raghad Shamma**  
**Dana Almounayer**  
Effat University  
Smart Recycling Bin – Deep Learning Project





