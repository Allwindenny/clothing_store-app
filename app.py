
from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import numpy as np
import joblib
from gensim.models import FastText
import uuid
import nltk
from nltk.stem import PorterStemmer


app = Flask(__name__)

# Load clothing data and models
df = pd.read_csv('assignment3_II.csv')  # Clothing data
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
lr_model = joblib.load('logistic_regression_model.pkl')
ft_model = FastText.load('fasttext_model.bin')

# Simulated database of reviews
reviews_db = {}

# Preprocess text for search and recommendation
def preprocess_text(text):
    return text.lower()

# Vectorize text using FastText (priority) and TF-IDF as fallback
def vectorize_text(text):
    text = preprocess_text(text)
    words = text.split()
    weighted_vectors = []

    for word in words:
        try:
            # Use FastText word vectors directly for each word
            word_vector = ft_model.wv[word]  # Get FastText word vector
            weighted_vectors.append(word_vector)
        except KeyError:
            # If the word is not in FastText vocabulary, skip it
            pass
    
    if len(weighted_vectors) > 0:
        # Return the average of all word vectors
        return sum(weighted_vectors) / len(weighted_vectors)
    else:
        # If no words match, return a zero vector
        return np.zeros(ft_model.vector_size)

# Predict recommendation based on the review description using FastText embeddings
def predict_recommendation(review_text, threshold=0.7):
    # Vectorize the review text using FastText
    review_vector = vectorize_text(review_text)
    review_vector = review_vector.reshape(1, -1)

    # Get prediction probabilities from the logistic regression model
    predicted_probability = lr_model.predict_proba(review_vector)[0][1]
    
    # Debugging print statements to check the model's output
    print(f"Review Text: {review_text}")
    print(f"Predicted Probability of Recommendation: {predicted_probability}")

    # Return recommendation based on the threshold
    if predicted_probability >= threshold:
        return 'Recommended'
    else:
        return 'Not Recommended'



# Initialize the Porter stemmer
stemmer = PorterStemmer()

# Download nltk resources if not already installed
nltk.download('punkt')

# Home route (search page)
'''@app.route('/')
def home():
    items = df[['Clothes Title', 'Clothes Description', 'Department Name', 'Division Name']].to_dict(orient='records')
    return render_template('home.html', items=items)'''
# Home route (search page)
@app.route('/')
def home():
    # Remove duplicate rows based on "Clothes Title"
    #items = df[['Clothes Title', 'Clothes Description', 'Department Name', 'Division Name']].drop_duplicates('Clothes Title').to_dict(orient='records')
    return render_template('home.html')


# Search route for searching clothing items with support for plural/non-plural forms
@app.route('/search', methods=['GET'])
def search():
    keyword = request.args.get('keyword', '').strip().lower()

    if keyword:
        print(f"Searching for keyword: {keyword}")
        
        # Stem the keyword (to handle plural/non-plural forms)
        stemmed_keyword = stemmer.stem(keyword)
        
        # Perform keyword search with stemming for similar forms in the specified fields
        search_results = df[df.apply(lambda row:
            stemmed_keyword in stemmer.stem(str(row['Clothes Title']).lower()) or
            stemmed_keyword in stemmer.stem(str(row['Clothes Description']).lower()) or
            stemmed_keyword in stemmer.stem(str(row['Department Name']).lower()) or
            stemmed_keyword in stemmer.stem(str(row['Division Name']).lower()), axis=1)]

        match_count = len(search_results)
        print(f"Found {match_count} matches")

        return render_template('search.html', results=search_results.to_dict(orient='records'), keyword=keyword, match_count=match_count)

    else:
        return render_template('search.html', results=[], keyword="", match_count=0)

@app.route('/reviews/<clothes_title>', methods=['GET'])#trying to view existing review
def view_reviews(clothes_title):
    # Filter the dataframe to get all reviews for the given clothes_title
    reviews = df[df['Clothes Title'] == clothes_title][['Review Text', 'Rating']].to_dict(orient='records')
    
    # If no reviews are found, set an empty list
    if not reviews:
        reviews = []

    return render_template('view_reviews.html', clothes_title=clothes_title, reviews=reviews)
# Route to view full details of an item
@app.route('/item/<clothing_id>')
def view_item(clothing_id):
    item = df[df['Clothing ID'] == int(clothing_id)].to_dict(orient='records')[0]
    reviews = [review for review in reviews_db.values() if review['clothing_id'] == clothing_id]
    return render_template('view_review.html', item=item, reviews=reviews)


# Route to add a new review for an item
@app.route('/add_review/<clothing_id>', methods=['GET', 'POST'])
def add_review(clothing_id):
    if request.method == 'POST':
        review_title = request.form['review_title']
        review_description = request.form['review_description']
        rating = request.form['rating']  # Capture the rating value

        review_text = f"{review_title} {review_description}"

        # Predict recommendation based on the review
        model_recommendation = predict_recommendation(review_text)

        # Pass the rating to the review form
        return render_template('review_form.html', review_text=review_text, model_recommendation=model_recommendation, clothing_id=clothing_id, rating=rating)
    
    clothing_item = df[df['Clothing ID'] == int(clothing_id)].to_dict(orient='records')[0]
    return render_template('add_review.html', clothing_item=clothing_item)

# Route to confirm review submission
@app.route('/confirm_review', methods=['POST'])
def confirm_review():
    final_recommendation = request.form['final_recommendation']
    review_text = request.form['review_text']
    rating = request.form['rating']  # Capture the rating
    clothing_id = request.form['clothing_id']
    
    review_id = str(uuid.uuid4())
    reviews_db[review_id] = {
        'clothing_id': clothing_id,
        'review_text': review_text,
        'recommendation': final_recommendation,
        'rating': rating  # Store rating with the review
    }
    
    return redirect(url_for('view_review', review_id=review_id))

# Route to view a submitted review
@app.route('/review/<review_id>')
def view_review(review_id):
    review = reviews_db.get(review_id)
    if review:
        return render_template('review_result.html', review=review)
    else:
        return "Review not found", 404



if __name__ == '__main__':
    app.run(debug=True)
