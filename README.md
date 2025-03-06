# 📚 Book Recommender

A machine learning-powered book recommendation system utilizing metadata from thousands of books. This project leverages natural language processing (NLP) and modern AI techniques to suggest books based on user preferences.

## 📂 Dataset
We use a comprehensive dataset containing metadata for 7,000+ books, sourced from Kaggle:
[7K Books with Metadata](https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata)

## 🛠 Installation
To set up the project, ensure you have Python installed, then install the required dependencies:

```sh
pip install kagglehub numpy pandas matplotlib seaborn langchain-openai \
            langchain-huggingface langchain-community langchain-chroma gradio \
            transformers notebook ipywidgets
```

## 🔧 Technologies Used
- **Data Processing:** `numpy`, `pandas`
- **Visualization:** `matplotlib`, `seaborn`
- **AI & NLP:** `transformers` (from Hugging Face), `langchain-openai`, `langchain-gemini`
- **Database for Storage:** `langchain-chroma`
- **Interactive UI:** `gradio`
- **Notebook Support:** `notebook`, `ipywidgets`
- **Frontend Interface:** `Gradio` for displaying the project and interacting with recommendations
- **Data Processing:** `numpy`, `pandas`
- **Visualization:** `matplotlib`, `seaborn`
- **AI & NLP:** `transformers` (from Hugging Face), `langchain-openai`, `langchain-gemini`
- **Database for Storage:** `langchain-chroma`
- **Interactive UI:** `gradio`
- **Notebook Support:** `notebook`, `ipywidgets`

## 🧠 Theory Behind the Project

### 😊 Sentiment Analysis
To enhance book recommendations, we incorporate **sentiment analysis** on user reviews. By analyzing emotions expressed in book reviews, we refine recommendations based on reader sentiment. We use a fine-tuned **RoBERTa model** from Hugging Face:
- **Model:** `j-hartmann/emotion-english-distilroberta-base`
- **Purpose:** Detect emotions in text (e.g., joy, sadness, anger, surprise, etc.)
- **Impact:** Helps recommend books that align with users' emotional preferences

### 🔤 Word Embeddings
Word embeddings are numerical vector representations of words in a continuous vector space. They allow words with similar meanings to have similar representations, making them essential for NLP tasks. We use **pre-trained transformer-based embeddings** to convert book metadata into vector representations, enabling efficient similarity comparisons in our recommendation engine.

Popular word embedding techniques include:
- **Word2Vec** (Mikolov et al.)
- **GloVe** (Pennington et al.)
- **BERT Embeddings** (Context-aware)

For this project, we leverage **transformer-based embeddings** from `Hugging Face`, specifically the **`sentence-transformers/all-MiniLM` model**, to create high-quality vector representations of books.
Word embeddings are numerical vector representations of words in a continuous vector space. They allow words with similar meanings to have similar representations, making them essential for NLP tasks. We use **pre-trained transformer-based embeddings** to convert book metadata into vector representations, enabling efficient similarity comparisons in our recommendation engine.

### 🤖 Transformers
Transformers are a deep learning architecture designed for NLP tasks. They use self-attention mechanisms to process text efficiently and capture contextual meaning. We utilize transformer-based models (like BERT, GPT, or OpenAI embeddings) to generate rich representations of book metadata, improving recommendation accuracy.

Key features of transformers:
- **Self-Attention Mechanism**: Helps models focus on important parts of text
- **Bidirectional Context Understanding**: Captures meaning from both left and right context
- **Scalability**: Suitable for large-scale NLP tasks

### 🏷️ Zero-Shot Classification for Categorization
Since book categories can be vast and dynamic, we employ **zero-shot classification** using transformer models. This allows us to classify books into predefined genres **without labeled training data**. By using models like `facebook/bart-large-mnli`, we match book descriptions with relevant genres on-the-fly, making the system flexible and adaptive.

## 🚀 Features
- 📖 **Personalized Book Recommendations** based on metadata and AI-driven insights
- 🔍 **Search and Filter Books** using natural language queries
- 🎨 **User-Friendly Interface** powered by Gradio
- 📊 **Data Visualization** for book insights and trends

## 🔮 Future Enhancements
- 🔗 Integration with external book APIs for enriched recommendations
- 📈 Improving the recommendation engine using deep learning
- 🌐 Deploying as a web application for wider accessibility

## 🤝 Contributing
Feel free to fork this repository, create a new branch, and submit a pull request with improvements or new features!

## 📜 License
This project is open-source and available under the MIT License.

---
⭐ If you like this project, give it a star on GitHub!
