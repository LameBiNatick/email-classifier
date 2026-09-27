# Email Classifier

### An ML-based project for classifying emails into categories

This project is a machine learning-based email classifier built with Python and Streamlit. It's designed to automatically categorize emails into one of four classes: **Spam**, **Promotions**, **Updates**, or **Personal**. The classifier uses TF-IDF for text vectorization and a Naive Bayes model for prediction, all wrapped in an interactive web application for real-time classification.

---

### Features

* **Real-time Classification**: A user-friendly Streamlit web application allows you to paste an email and get an instant classification.
* **TF-IDF Vectorization**: Utilizes the efficient TF-IDF (Term Frequency-Inverse Document Frequency) algorithm to convert text data into numerical vectors.
* **Machine Learning Models**: The project trains and compares both Naive Bayes and Random Forest classifiers to find the best-performing model.
* **Model Persistence**: The trained machine learning model is saved using `joblib` to prevent retraining every time the application runs.
* **Data Generation**: Includes a script (`emails_expanded.py`) to generate synthetic email data for training purposes.
* **Word Cloud Visualization**: A word cloud is generated for each category to provide insights into the most frequent terms.

---

### Technologies Used

* **Python**: The core programming language for the project.
* **scikit-learn**: A robust library for machine learning model training and evaluation.
* **Streamlit**: The framework used to create the interactive web application.
* **`joblib`**: Used for saving and loading the trained machine learning model.
* **`wordcloud`**: A library for creating word cloud visualizations.

---

### Getting Started

Follow these steps to get a local copy of the project up and running on your machine.

#### Prerequisites

Make sure you have Python installed. This project uses Python 3.8 or higher.

#### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/JCAR4/Email-classifier.git](https://github.com/JCAR4/Email-classifier.git)
    cd Email-classifier
    ```

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Train the model:**
    Run the training script to generate the trained model (`classifier.joblib`).
    ```bash
    python train.py
    ```

5.  **Run the application:**
    Launch the Streamlit web application to start classifying emails.
    ```bash
    streamlit run app.py
    ```
    The application will open in your default web browser.

---

### Model Performance

The model's performance was evaluated based on the following metrics:

* **Accuracy:** [Add your model's accuracy score here, e.g., 95.8%]
* **Classification Report:** A detailed report on precision, recall, and F1-score for each category.

---

### License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

---

