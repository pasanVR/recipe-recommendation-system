import numpy as np
import pickle
import streamlit as st
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import os
import pandas as pd
import functions

# Define the functions
def find_image(recipe_title, image_dir):
    # Convert recipe title to lowercase and replace spaces with hyphens, and remove apostrophes
    recipe_name = recipe_title.lower().replace(' ', '-').replace("'", "")
    
    # Try to find an image without digits appended
    image_path = os.path.join(image_dir, f"{recipe_name}.jpg")
    if os.path.exists(image_path):
        return image_path
    
    # If not found, try to find an image with digits appended
    for filename in os.listdir(image_dir):
        if recipe_name in filename.lower() and filename.lower().endswith('.jpg'):
            return os.path.join(image_dir, filename)
    
    # If still not found, return None
    return None

def format_ingredients(ingredients):
    # Remove square brackets, replace commas with newlines
    return ingredients.replace('[', '').replace(']', '').replace(', ', '\n')

def recommend(recipe_title, data, ct_vectorizer, lsa_10, ct_lsa_10, image_dir):
    # Find the index of the selected recipe
    index = data[data['Title'] == recipe_title].index[0]
    
    # Compute similarity scores for the selected recipe
    selected_recipe_vector = ct_lsa_10[index].reshape(1, -1)
    scores = cosine_similarity(selected_recipe_vector, ct_lsa_10)
    
    # Get top 5 similar recipes
    distance = sorted(list(enumerate(scores[0])), reverse=True, key=lambda x: x[1])
    recommendations = []
    for idx in distance[1:6]:  # Skip the first one as it is the same recipe
        recipe_data = data.iloc[idx[0]]
        image_path = find_image(recipe_data['Title'], image_dir)
        recommendations.append({
            'Title': recipe_data['Title'],
            'Ingredients': format_ingredients(recipe_data['Ingredients']),
            'Instructions': recipe_data['Instructions'],
            'image_path': image_path
        })
    return recommendations

def recommend_using_ingredients(user_input, data, ct_vectorizer, lsa_10, ct_lsa_10, image_dir):
    # Process the input ingredients
    input_ingr = functions.regex_nodigits_new(user_input)
    input_vector = ct_vectorizer.transform([input_ingr])
    input_lsa = lsa_10.transform(input_vector)
    
    # Compute similarity scores
    scores = cosine_similarity(input_lsa, ct_lsa_10)
    
    # Get top 5 similar recipes
    indices = np.argsort(scores[0])[::-1][1:6]
    
    recommendations = []
    for idx in indices:
        recipe_data = data.iloc[idx]
        image_path = find_image(recipe_data['Title'], image_dir)
        recipe_lsa = lsa_10.transform(ct_vectorizer.transform([recipe_data['CleanIngredients']]))
        match_percentage = cosine_similarity(input_lsa, recipe_lsa)[0][0] * 100
        recommendations.append({
            'Title': recipe_data['Title'],
            'Ingredients': format_ingredients(recipe_data['Ingredients']),
            'Instructions': recipe_data['Instructions'],
            'image_path': image_path,
            'match_percentage': match_percentage
        })
    return recommendations

# Load the models and data for all recipes
all_recipes = pickle.load(open('Artifacts/RecipeList.pkl', 'rb'))
all_ct_vectorizer = pickle.load(open('Artifacts/ct_vectorizer.pkl', 'rb'))
all_lsa_10 = pickle.load(open('Artifacts/lsa_10.pkl', 'rb'))
all_ct_lsa_10 = pickle.load(open('Artifacts/ct_lsa_10.pkl', 'rb'))

# Load the models and data for Sri Lankan recipes
srilankan_recipes = pickle.load(open('Artifacts/sl_RecipeList.pkl', 'rb'))
sl_ct_vectorizer = pickle.load(open('Artifacts/sl_ct_vectorizer.pkl', 'rb'))
sl_lsa_10 = pickle.load(open('Artifacts/sl_lsa_10.pkl', 'rb'))
sl_ct_lsa_10 = pickle.load(open('Artifacts/sl_ct_lsa_10.pkl', 'rb'))

# Directory where recipe images are stored
image_dir = 'Food_Images/'  # Update this path to your image directory

# Streamlit interface
st.header("Recipe Recommendation System")



# Ingredient-based recommendation
st.subheader("Recommend recipes based on ingredients")
user_input = st.text_input("Enter ingredients separated by commas", key='ingredient_input')
# Dropdown to select dataset
dataset_option = st.selectbox('Select the recipe dataset', ('All Recipes', 'Sri Lankan Recipes'))

# Determine the dataset based on selection
if dataset_option == 'All Recipes':
    data = all_recipes
    ct_vectorizer = all_ct_vectorizer
    lsa_10 = all_lsa_10
    ct_lsa_10 = all_ct_lsa_10
else:
    data = srilankan_recipes
    ct_vectorizer = sl_ct_vectorizer
    lsa_10 = sl_lsa_10
    ct_lsa_10 = sl_ct_lsa_10

if st.button('Recommend', key='recommend_button'):
    recommendations = recommend_using_ingredients(user_input, data, ct_vectorizer, lsa_10, ct_lsa_10, image_dir)
    for recipe in recommendations:
        st.text(f"Recipe Title: {recipe['Title']}")
        if recipe['image_path']:
            image = Image.open(recipe['image_path'])
            st.image(image, caption=recipe['Title'], use_column_width=True)
        st.text(f"Ingredients:\n{recipe['Ingredients']}")
        st.text_area(f"Instructions for {recipe['Title']}:", recipe['Instructions'], height=200)
        st.text(f"Match Percentage: {recipe['match_percentage']:.2f}%")
        st.text("_________________________________________________")

# Recipe-based recommendation
st.subheader("Recommend based on a recipe")
recipe_list = data['Title'].values
selected_recipe = st.selectbox(
    'Type or select a recipe to get recommendation',
    recipe_list,
    key='recipe_selectbox'
)

if st.button('Show recommendation', key='show_recommendation_button'):
    recommendations = recommend(selected_recipe, data, ct_vectorizer, lsa_10, ct_lsa_10, image_dir)
    for recipe in recommendations:
        st.text(f"Recipe Title: {recipe['Title']}")
        if recipe['image_path']:
            image = Image.open(recipe['image_path'])
            st.image(image, caption=recipe['Title'], use_column_width=True)
        st.text(f"Ingredients:\n{recipe['Ingredients']}")
        st.text_area(f"Instructions for {recipe['Title']}:", recipe['Instructions'], height=200)
        st.text("_________________________________________________")
