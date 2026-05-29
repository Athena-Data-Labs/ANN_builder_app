# Neural Network Studio App

Build and train artificial neural networks with an intuitive Streamlit GUI. Easily upload and explore your data, fine-tune model parameters, and visualize both your data and model performance—all in one streamlined app.

👉 **[Launch the app](https://ann-builder-app.streamlit.app)**


---

## 🚀 Features

- **📂 Data Upload:** Easily upload your CSV dataset for analysis.
- **🧹 Data Cleaning & Preprocessing:** Clean missing values, remove duplicates, and prepare your data for modeling before training.
- **📊 Data Exploration & Visualization:**  
  - View summary statistics, missing values, and correlation matrices.
  - Generate interactive charts: area, box, histogram, scatter, heatmap, and pie.
- **🏗️ Build Artificial-Neural-Network:**  
  - Configure, train, and evaluate an ANN model on your data.
  - Adjust hyperparameters and monitor training progress.
  - Upload new (unseen) data to make predictions with your trained model.
  - Download prediction results as a CSV file.
- **🤖 About:** Learn what each page does and how to use the app.

---

![Example Neural Network Visualization](reports/figures/created_ANN.png)
*Example: Visualizing a generated neural network with the app.*

---

## 🗂️ Project Organization

```
├── LICENSE
├── README.md
├── data
│   ├── external
│   ├── interim
│   ├── processed
│   └── raw
├── docs
├── models
├── notebooks
├── pyproject.toml
├── references
├── reports
│   └── figures
├── requirements.txt
└── src
    ├── __init__.py
    ├── about.py
    ├── config.py
    ├── data_exploration.py
    ├── data_preprocessing.py
    ├── dataset.py
    ├── features.py
    ├── main.py
    ├── model_build.py
    ├── visualization
    │   ├── __init__.py
    │   └── visualize.py
    └── modeling
        ├── __init__.py
        ├── predict.py
        └── train.py
```

---

## ⚡ Getting Started

1. **Install requirements:**  
   ```
   pip install -r requirements.txt
   ```

2. **Run the app:**  
   ```
   streamlit run src/main.py
   ```

3. **Open in your browser:**  
   Visit [http://localhost:8501](http://localhost:8501) (default Streamlit port).

The app now includes a working preprocessing page and is deployed from `src/main.py`.

---

## 📄 License

See [LICENSE](LICENSE) for details.

---

**Author:** Vahidin (Dean) Jupic  
**Version:** 0.9.0
