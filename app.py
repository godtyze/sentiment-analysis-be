from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS
from  youtube_component import get_youtube_comments
from flask_sqlalchemy import SQLAlchemy
from flask import send_file
from io import BytesIO

app = Flask(__name__)
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////instance/feedback.db'
# db = SQLAlchemy(app)
CORS(app)

# class UserFeedback(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     original_text = db.Column(db.String, nullable=False)
#     user_sentiment = db.Column(db.String, nullable=False)
#     user_comment = db.Column(db.String)
#
# with app.app_context():
#     db.create_all()

# Load your trained model and vectorizer
model = joblib.load('trained_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
@app.route('/analyze-text', methods=['POST'])
def analyze_text():
    data = request.get_json()
    print("Received data:", data)
    text = data.get('text')

    if not isinstance(text, str):
        return jsonify({'error': 'Text data is not a string'}), 400

    # Vectorize the text
    processed_text = vectorizer.transform([text])

    # Predict sentiment
    prediction = model.predict(processed_text)[0]
    sentiment = 'Positive' if prediction == "1" else 'Negative'

    return jsonify({'text': text, 'Sentiment': sentiment})
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

        processed_texts = vectorizer.transform(df[column_name])
        predictions = model.predict(processed_texts)
        df['Sentiment'] = ['Positive' if pred == "1" else 'Negative' for pred in predictions]

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
    processed_texts = vectorizer.transform(comments_df['text'])
    predictions = model.predict(processed_texts)
    
    # Add the predictions to the DataFrame
    comments_df['Sentiment'] = ['Positive' if pred == "1" else 'Negative' for pred in predictions]

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

# @app.route('/submit-feedback', methods=['POST'])
# def submit_feedback():
#     data = request.get_json()
#     original_text = data.get('original_text')
#     user_sentiment = data.get('user_sentiment')
#     user_comment = data.get('user_comment')
#
#     feedback = UserFeedback(original_text=original_text, user_sentiment=user_sentiment, user_comment=user_comment)
#     db.session.add(feedback)
#     db.session.commit()
#
#     return jsonify({'message': 'Feedback submitted successfully'}), 200

# @app.route('/get-feedback', methods=['GET'])
# def get_feedback():
#     try:
#         # Query all feedback records
#         feedback_records = UserFeedback.query.all()
#
#         # Convert records to a list of dictionaries
#         feedback_list = [{"id": feedback.id,
#                           "original_text": feedback.original_text,
#                           "user_sentiment": feedback.user_sentiment,
#                           "user_comment": feedback.user_comment}
#                          for feedback in feedback_records]
#
#         return jsonify(feedback_list)
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500
#
# @app.route('/delete-feedback/<int:feedback_id>', methods=['DELETE'])
# def delete_feedback(feedback_id):
#     try:
#         # Find the feedback record by ID
#         feedback = UserFeedback.query.get(feedback_id)
#         if not feedback:
#             return jsonify({'error': 'Feedback not found'}), 404
#
#         # Delete the record
#         db.session.delete(feedback)
#         db.session.commit()
#
#         return jsonify({'message': 'Feedback deleted successfully'}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

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
    serve(app, host="localhost", port=8080)
