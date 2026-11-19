# 🌾 Machine Learning Based Crop Yield Prediction & Recommendation System

A comprehensive machine learning web application that predicts agricultural crop yields and production based on state, district, crop type, season, and cultivated area, and provides multi-criteria crop recommendations. Built with Flask, scikit-learn, and XGBoost, this application helps farmers and agricultural planners make data-driven decisions.

## ✨ Features

- **Smart Prediction**: Uses Random Forest machine learning algorithm for accurate predictions
- **User-Friendly Interface**: Modern, responsive web design with dropdown selections
- **Comprehensive Data**: Trained on diverse crop production data from Indian states
- **Real-time Results**: Instant production estimates in tons
- **Mobile Responsive**: Works seamlessly on desktop and mobile devices

## 🚀 Quick Start

### Prerequisites
- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. **Clone or download the project**
   ```bash
   # If you have git
   git clone <repository-url>
   cd crop-production-predictor
   
   # Or simply navigate to the project folder
   cd crop-production-predictor
   ```

2. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python src/app.py
   ```

4. **Open your browser**
   Navigate to `http://localhost:5000`

## 📱 How to Use

1. **Select State**: Choose from 16 Indian states including Maharashtra, Punjab, Karnataka, etc.
2. **Choose Crop**: Select from 25+ crop types including Rice, Wheat, Cotton, Sugarcane, etc.
3. **Enter Area**: Input the area in acres (1-10,000 acres)
4. **Get Prediction**: Click "🔮 Predict Production" to get your estimated production in tons

## 🎯 Supported States

- Andhra Pradesh
- Bihar
- Gujarat
- Haryana
- Karnataka
- Kerala
- Madhya Pradesh
- Maharashtra
- Odisha
- Punjab
- Rajasthan
- Tamil Nadu
- Telangana
- Uttar Pradesh
- West Bengal

## 🌱 Supported Crops

- **Cereals**: Rice, Wheat, Corn, Maize, Barley, Millets, Bajra
- **Pulses**: Chickpeas, Pulses
- **Oilseeds**: Mustard, Groundnut, Oilseeds, Soybean
- **Cash Crops**: Cotton, Sugarcane, Tobacco, Tea, Coffee
- **Spices**: Chili, Spices
- **Others**: Fruits, Vegetables, Jute, Rubber, Coconut

## 🏗️ Project Structure

```
crop-production-predictor/
├── src/
│   ├── app.py                 # Main Flask application
│   ├── models/
│   │   └── predictor.py       # Machine learning model
│   ├── routes/
│   │   └── predict.py         # API routes
│   ├── templates/
│   │   └── index.html         # Web interface
│   └── static/
│       └── style.css          # Styling
├── data/
│   └── crop_data.csv          # Training dataset
├── requirements.txt           # Python dependencies
├── test_app.py               # Test script
└── README.md                 # This file
```

## 🔧 Technical Details

### Machine Learning Model
- **Algorithm**: Random Forest Regressor
- **Features**: State, Crop Type, Area
- **Target**: Production (in tons)
- **Preprocessing**: One-hot encoding for categorical variables
- **Validation**: 80/20 train-test split

### Web Framework
- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript
- **Styling**: Custom CSS with gradient backgrounds and animations
- **Responsive**: Mobile-first design approach

## 🧪 Testing

Run the test script to verify everything works:

```bash
python test_app.py
```

This will test:
- ✅ All imports work correctly
- ✅ Machine learning model trains successfully
- ✅ Predictions are generated
- ✅ Flask application starts properly

## 📊 Sample Predictions

| State | Crop | Area (acres) | Predicted Production (tons) |
|-------|------|--------------|----------------------------|
| Maharashtra | Wheat | 100 | ~2,000 |
| Punjab | Rice | 150 | ~3,000 |
| Karnataka | Sugarcane | 200 | ~4,000 |

## 🛠️ Customization

### Adding New States/Crops
1. Update `data/crop_data.csv` with new data
2. Add options to the dropdowns in `src/templates/index.html`
3. Retrain the model by restarting the application

### Modifying the Model
Edit `src/models/predictor.py` to:
- Change the machine learning algorithm
- Add new features
- Modify preprocessing steps

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure you're in the correct directory
   - Check that all dependencies are installed

2. **Prediction Errors**
   - Make sure the state and crop names match exactly
   - Check that area is a positive number

3. **Flask App Won't Start**
   - Verify port 5000 is available
   - Check for syntax errors in the code

### Getting Help

If you encounter issues:
1. Run `python test_app.py` to identify problems
2. Check the console output for error messages
3. Ensure all files are in the correct locations

## 📈 Future Enhancements

- [ ] Add weather data integration
- [ ] Implement user authentication
- [ ] Add historical data visualization
- [ ] Support for more countries/regions
- [ ] Export predictions to PDF/Excel
- [ ] Mobile app version

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Data sourced from various agricultural databases
- Built with Python, Flask, and scikit-learn
- UI inspired by modern agricultural applications

---

**Happy Farming! 🌾**