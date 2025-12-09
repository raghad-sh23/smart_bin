# Smart Bin – Waste Classification Using Deep Learning

This project is an AI-powered Smart Recycling Bin that classifies waste into five categories:
- Plastic  
- Paper  
- Glass  
- Metal  
- General Waste  

It uses TensorFlow/Keras with transfer learning (MobileNetV2) and includes a prediction interface for testing individual images.

---

## 📌 Features
- Transfer Learning using MobileNetV2  
- Model training script  
- Real-time classification interface (`smart_bin_app.py`)  
- Class map generation  
- Training history visualization  
- Pretrained model files included  

---

## 🧠 Model Files
The repository includes two trained models:

- `models/smart_bin_mbv2.h5`  
- `models/smart_bin_mbv2_best.keras`  

You can load them in the app with:

```python
model = load_model("models/smart_bin_mbv2_best.keras")
