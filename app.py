from flask import Flask, render_template, request, redirect, url_for, session
import pickle
app = Flask(__name__)
app.secret_key = '123'  # Set a secret key for session management


# Load your recommendation system models and data from the pickle file here.
with open('book_recommendation.pkl', 'rb') as file:
    data = pickle.load(file)
    model_knn1 = data['model_knn1']
    user_features_df = data['user_features_df']
    rating_popular_books_df = data['rating_popular_books_df']
    model_knn2 = data['model_knn2']
    book_features_df = data['book_features_df']


def recommenduserbased(user_id=10):
    # Your existing user-based recommendation code here
    # ...
    n_users = 5
    rec_top_n = 10
    distances, indices = model_knn1.kneighbors(user_features_df.loc[user_features_df.index == user_id].values.reshape(1, -1), n_neighbors=n_users + 1)
    user_ids = []
    recommended_titles = []
    for index in range(0, len(distances.flatten())):
        user_ids.append(user_features_df.index[indices.flatten()[index]])
        
    # select books that were highly ranked by the most similar users.

    # look only for books highly rated by the similar users, not the current user
    candidate_user_ids = user_ids[1:]
    sel_ratings = rating_popular_books_df.loc[rating_popular_books_df['User-ID'].isin(candidate_user_ids)]
    # sort by best ratings and total rating count
    sel_ratings = sel_ratings.sort_values(by=["Book-Rating", "total_rating_count"], ascending=False)
    # eliminate from the selection books that were ranked already by the current user
    books_rated_by_targeted_user = list(rating_popular_books_df.loc[rating_popular_books_df['User-ID'] == user_ids[0]]["ISBN"].values)
    sel_ratings = sel_ratings.loc[~sel_ratings['ISBN'].isin(books_rated_by_targeted_user)]
    # aggregate and count total ratings and total total_rating_count
    agg_sel_ratings = sel_ratings.groupby(["Book-Title", "Book-Rating"])["total_rating_count"].max().reset_index()
    agg_sel_ratings.columns = ["Book-Title", "Book-Rating", "total_ratings"]
    agg_sel_ratings = agg_sel_ratings.sort_values(by=["Book-Rating", "total_ratings"], ascending=False)
    # only select top n (default top 10 here)
    recommended_titles = agg_sel_ratings["Book-Title"].head(10).values
    '''rec_list = agg_sel_ratings["Book-Title"].head(10).values
    recommendations = [f"{i+1}: {rec}" for i, rec in enumerate(rec_list)]'''
    
    return recommended_titles


# Define the recommend_item_based function
def recommend_item_based(book_title, top_n=10):
    # Your existing item-based recommendation code here
    # ...
    # Check if the book_title exists in the dataset
    if book_title not in book_features_df.index:
        return []  # Return an empty list if the movie is not found

    # Find the query_index for the given book_title
    query_index = book_features_df.index.get_loc(book_title)

    distances, indices = model_knn2.kneighbors(book_features_df.iloc[query_index, :].values.reshape(1, -1), n_neighbors=top_n + 1)

    recommended_books = []
    for index in range(1, len(distances.flatten())):  # Start from 1 to exclude the input book
        recommended_books.append(book_features_df.index[indices.flatten()[index]])

    return recommended_books

# Define routes and their functionality
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend')
def recommend():
    
    return render_template('input.html')

@app.route('/user_recommendtions', methods=['POST', 'GET'])
def user_recommendations():
    # Retrieve the user_id from the session
    user_id = request.form['user_id']  # Use 10 as the default value if user_id is not found in the session
    if not user_id.isdigit():
        error_message1 = 'please enter only numeric values for User ID.'
    if error_message1:
        return render_template('input.html', error_message1=error_message1)
    user_id = int(user_id)
    recommended_books1 = recommenduserbased(user_id)
    return render_template('user_recommendations.html', user_id=user_id, recommended_books=recommended_books1,error_message1=error_message1)

@app.route('/book_recommendations', methods=['POST'])
def book_recommendations():
    book_title = request.form['book_title']
    error_message2 = None  # Initialize error_message

    if book_title not in book_features_df.index:
        error_message2 = f'The book "{book_title}" does not exist.'

    if error_message2:
        # If there is an error, re-render the form with the error message
        return render_template('input.html', error_message2=error_message2)

    # If no error, proceed with movie recommendations
    recommended_books2 = recommend_item_based(book_title)
    return render_template('book_recommendations.html', book_title=book_title, recommended_books=recommended_books2, error_message2=error_message2)

if __name__ == '__main__':
    app.run(debug=True)



