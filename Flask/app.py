from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load('best_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Define the label mapping based on your model's output
label_mapping = {
    1: "Hate Speech",
    2: "Normal Speech"
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = tfidf_vectorizer.transform(data).toarray()
        prediction = model.predict(vect)
        prediction_label = label_mapping.get(prediction[0], "Unknown")
        return render_template('index.html', prediction=prediction_label)

if __name__ == '__main__':
    app.run(debug=True)
