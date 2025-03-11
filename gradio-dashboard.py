import numpy as np
import pandas as pd
import os

from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import gradio as gr
load_dotenv()

books = pd.read_csv("books_with_emotions.csv")

#we want google book to return us the largest thumbnail
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(books["large_thumbnail"].isna(), "assets/cover-not-found.jpg", books["large_thumbnail"])


raw_documents = TextLoader("tagged_description.txt").load()
text_splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator="\n")
documents = text_splitter.split_documents(raw_documents)
huggingface_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")



# make the chroma database so that the model doesn't need to run every time
DB_DIR = "chroma_db"
os.makedirs(DB_DIR, exist_ok=True)

if os.path.exists(DB_DIR) and len(os.listdir(DB_DIR)) > 0:
    print("Loading existing Chroma database...")
    db_books = Chroma(persist_directory=DB_DIR, embedding_function=huggingface_embeddings)
else:
    print("Creating new Chroma database...")
    db_books = Chroma.from_documents(documents, huggingface_embeddings, persist_directory=DB_DIR)




def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
)->pd.DataFrame:
    recs = db_books.similarity_search(query, k = initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books['isbn13'].isin(books_list)].head(final_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs





#now we will create a function about what we want to display in our gradio dashboard

def recommend_books(
        query: str,
        category: str = None,
        tone: str = None,
):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, rec in recommendations.iterrows():
        description = rec["description"]
        truncate_desc_split = description.split()
        truncate_description = " ".join(truncate_desc_split[:30]) + "..."

        authors_split = rec["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f" {authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = rec["authors"]

        caption =f"{rec['title']} by {authors_str} : {truncate_description}"
        results.append((rec["large_thumbnail"], caption))

    return results




#Gradio

categories = ["All"] + sorted(books['simple_categories'].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme= gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic Book Recommender")

    with gr.Row():
        user_query = gr.Text(label= "Please enter the description of the book:",
                             placeholder="e.g., A story about ghost king")
        category_dropdown = gr.Dropdown(choices = categories, label = "Select a Category", value = "All")
        tone_dropdown = gr.Dropdown(choices = tones, label = "Select an emotion", value = "All")
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label="Recommended Books", columns = 8, rows = 2)
    submit_button.click(fn = recommend_books, inputs = [user_query, category_dropdown, tone_dropdown], outputs = output)




if __name__ == "__main__":
    dashboard.launch()


