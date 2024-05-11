from flask import Flask, request, jsonify
import pandas as pd
from flask_cors import CORS
from youtube_component import get_youtube_comments
from flask import send_file
from io import BytesIO
from transformers import pipeline

app = Flask(__name__)
CORS(app)

# Load your trained model and vectorizer
# model = joblib.load('trained_model.pkl')
# vectorizer = joblib.load('vectorizer.pkl')

model = pipeline(
    model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
    return_all_scores=True,
    truncation=True,
)


@app.route('/analyze-text', methods=['POST'])
def analyze_text():
    data = request.get_json()
    print("Received data:", data)
    text = data.get('text')

    if not isinstance(text, str):
        return jsonify({'error': 'Text data is not a string'}), 400

    prediction = model(text)[0]

    return jsonify({'text': text, 'sentiment': prediction})


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    column_name = request.form.get('column_name', 'text')  # Get column name from the request, default to 'text'

    # Read the file and process
    if file:
        df = pd.read_csv(file)
        if column_name not in df.columns:
            return jsonify({'error': f'Column {column_name} not found in the file'}), 400

        predictions = []

        for text in df[column_name]:
            predictions.append(model(text)[0])

        df['sentiment'] = [pred for pred in predictions]

        # Convert dataframe to a list of dictionaries for JSON response
        results = df.to_dict(orient='records')
        return jsonify(results)

    return jsonify({'error': 'No file selected'}), 400


@app.route('/analyze-youtube-comments', methods=['POST'])
def analyze_youtube_comments():
    data = request.get_json()
    video_id = data.get('video_id')

    if not video_id:
        return jsonify({'error': 'No video ID provided'}), 400

    comments_df = get_youtube_comments(video_id)
    predictions = []

    for text in comments_df['text']:
        if len(text) > 512:
            text = text[:512]

        predictions.append(model(text)[0])

    comments_df['sentiment'] = [pred for pred in predictions]

    # Convert dataframe to a list of dictionaries for JSON response
    results = comments_df.to_dict(orient='records')
    return jsonify(results)


@app.route('/calculate-sentiment-percentages', methods=['POST'])
def calculate_sentiment_percentages():
    data = request.get_json()
    sentiments = data.get('sentiments')

    if not sentiments or not isinstance(sentiments, list):
        return jsonify({'error': 'Invalid or missing sentiments data'}), 400

    sentiment_df = pd.DataFrame(sentiments, columns=['Sentiment'])
    sentiment_counts = sentiment_df['Sentiment'].value_counts(normalize=True) * 100
    sentiment_percentages = sentiment_counts.to_dict()

    return jsonify({'sentiment_percentages': sentiment_percentages})


@app.route('/export-csv', methods=['POST'])
def export_csv():
    data = request.get_json()
    results = data.get('results')
    buffer = BytesIO()

    if not results or not isinstance(results, list):
        return jsonify({'error': 'Invalid or missing results data'}), 400

    # Convert data to DataFrame and then to CSV
    try:
        df = pd.DataFrame(results)
        # Create a buffer to hold CSV data
        df.to_csv(buffer, index=False)
        buffer.seek(0)  # Rewind the buffer

        # Create a response object, sending the buffer as a file
        return send_file(
            buffer,
            as_attachment=True,
            download_name='results.csv',
            mimetype='text/csv'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    from waitress import serve

    serve(app, host="0.0.0.0", port=8080)
