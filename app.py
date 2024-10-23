import streamlit as st
import feedparser
import requests
import os
import json
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import openai
import faiss
import numpy as np
from openai import OpenAI
from typing import List
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores import FAISS
from langchain.schema import Document 
# Clé API OpenAI
OPENAI_API_KEY=st.secrets("OPENAI_API_KEY")
st.session_state.OPENAI_API_KEY=OPENAI_API_KEY
client = OpenAI(
api_key=OPENAI_API_KEY,
)
st.session_state.client = client
# Configuration de l'API OpenAI
openai.api_key = OPENAI_API_KEY

# URL du flux RSS
RSS_URL = "https://www.francetvinfo.fr/titres.rss"

# Fichier pour stocker les articles traités
PROCESSED_ARTICLES_FILE = "processed_articles.json"

# Définition de la classe Article
class Article:
    def __init__(self, title, link, content, chapo, published, summary=None):
        self.title = title
        self.link = link
        self.content = content
        self.chapo = chapo
        
        # Convertir published en datetime si c'est une chaîne
        if isinstance(published, str):
            self.published = datetime.fromisoformat(published)  # Conversion de la chaîne en datetime
        else:
            self.published = published  # Supposer que c'est déjà un objet datetime
        
        self.summary = summary
        self.metadata = {"title": self.title, "published": self.published.isoformat()}  # Ajout de metadata

    def __str__(self):
        return (f"Title: {self.title}\n"
                f"Link: {self.link}\n"
                f"Published: {self.published.strftime('%d %B %Y %H:%M')}\n"
                f"Chapo: {self.chapo}\n"
                f"Content: {self.content[:100]}...")  # Affiche les 100 premiers caractères du contenu

    @property
    def page_content(self):
        return self.summary  # Retourne le contenu de l'article comme page_content

    def to_dict(self):
        return {
            'title': self.title,
            'link': self.link,  # Inclure le lien dans le dictionnaire
            'chapo': self.chapo,
            'content': self.content,
            'published': self.published.isoformat(),
            'summary': self.summary,
            'metadata': self.metadata,  # Inclure metadata dans le dictionnaire
        }

    @staticmethod
    def from_dict(data):
        # Convertir les données en un objet Article
        published = datetime.fromisoformat(data['published'])
        return Article(data['title'], data['link'], data['chapo'], data['content'], published, data.get('summary', ""))

# Fonction pour charger les articles déjà traités depuis un fichier
def load_processed_articles():
    if os.path.exists(PROCESSED_ARTICLES_FILE):
        with open(PROCESSED_ARTICLES_FILE, 'r') as file:
            try:
                articles_data = json.load(file)
                # Convertir les dictionnaires en objets Article
                articles = [
                    Article(
                        title=article['title'],
                        link=article['link'],
                        content=article['content'],
                        chapo=article['chapo'],
                        published=article['published'],
                        summary=article.get('summary')
                    )
                    for article in articles_data
                ]
                return articles
            except json.JSONDecodeError:
                print("Le fichier est vide ou mal formé. Initialisation d'une liste vide.")
                return []
    else:
        # Créer le fichier avec une liste vide s'il n'existe pas
        with open(PROCESSED_ARTICLES_FILE, 'w') as file:
            json.dump([], file)  # Initialise avec une liste vide
        return []
    
# Fonction pour enregistrer les articles traités dans un fichier
def save_processed_articles(articles):
    with open(PROCESSED_ARTICLES_FILE, 'w') as file:
        json.dump([article.to_dict() for article in articles], file)

# Vérifiez si 'processed_links' est dans le session_state
if 'processed_links' not in st.session_state:
    st.session_state.processed_links = []  # Initialiser comme liste

# Charger les articles déjà traités
articles = load_processed_articles()

# Extraire les liens des articles traités et les stocker dans st.session_state.processed_links
st.session_state.processed_links = [article.link for article in articles]


# Fonction pour récupérer les articles du flux RSS
def get_rss_feed(url):
    feed = feedparser.parse(url)
    articles = []
    
    # Filtrer les articles des dernières 24h
    for entry in feed.entries:
        published_time = datetime(*entry.published_parsed[:6])
        if datetime.now() - published_time < timedelta(days=1):
            article = {
                "title": entry.title,
                "description": entry.description,
                "link": entry.link,
                "published": published_time
            }
            articles.append(article)
    return articles

def get_article_content(link):
    response = requests.get(link)
    soup = BeautifulSoup(response.content, "html.parser")
    
    # Extraire le titre
    title_element = soup.find("h1", class_="c-title")
    title = title_element.get_text(strip=True) if title_element else "Titre introuvable"
    
    # Extraire le chapo
    chapo_element = soup.find("div", class_="c-chapo")
    chapo = chapo_element.get_text(strip=True) if chapo_element else "Chapo introuvable"
    
    # Extraire le contenu de l'article
    content_div = soup.find("div", class_="c-body")
    content = content_div.get_text(separator="\n").strip() if content_div else "Contenu introuvable"

    # Extraire la date de publication
    date_div = soup.find("div", class_="publication-date")
    published_time = None
    if date_div:
        time_element = date_div.find("time")
        if time_element and time_element.has_attr("datetime"):
            published_time = datetime.fromisoformat(time_element["datetime"])
        else:
            published_time = datetime.now()  # Date actuelle si introuvable
    else:
        published_time = datetime.now()  # Date actuelle si introuvable
    
    # Créer une instance de l'article avec un résumé vide par défaut
    article = Article(title, link, chapo, content, published_time, summary="")  # Ajout d'un résumé vide
    return article

# Fonction pour générer un résumé via OpenAI
def generate_summary(article_content):
    client = st.session_state.client
    response = client.chat.completions.create(
        messages=[{
            "role": "user",
            "content": f"Fais un résumé des points clés suivants :\n\n{article_content}",
        }],
        model="gpt-3.5-turbo",
    )
    summary = response.choices[0].message.content
    return summary

# Fonction pour splitter les documents
def load_and_split_documents(documents, chunk_size=100, chunk_overlap=10):
    text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = text_splitter.split_documents(documents)
    docs_content = "\n\n".join([doc.page_content for doc in split_docs])
    print(docs_content)
    st.write(f"Nombre de chunks générés : {len(split_docs)}")  # Passer une liste d'objets Article
    return split_docs

def store_embeddings(docs):
    embeddings = OpenAIEmbeddings(openai_api_key=st.session_state.OPENAI_API_KEY)

    # Vérifier si l'index FAISS existe déjà
    if os.path.exists("faiss_index"):
        # Charger l'index existant
        vectordb = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

        # Ajouter tous les documents à la base de données FAISS
        vectordb.add_documents(docs)  # Passer la liste de documents ici
    else:
        # Si l'index n'existe pas, créer un nouveau vectordb
        vectordb = FAISS.from_documents(docs, embeddings)

    # Sauvegarder l'index mis à jour
    vectordb.save_local("faiss_index")

    # Mettre à jour l'état de la session avec le vectordb
    st.session_state.vectordb = vectordb

# Fonction pour afficher les articles avec Streamlit
def display_articles(articles):
    st.title("Actualités des dernières 24 heures")

    # Charger les articles déjà traités (retourne une liste complète d'articles)
    stored_articles = load_processed_articles()

    # Extraire les liens des articles déjà traités pour éviter les doublons
    if 'processed_links' not in st.session_state:
        st.session_state.processed_links = [article['link'] for article in stored_articles if 'link' in article]

    for article in articles:
        st.header(article['title'])
        st.write(f"*Publié le :* {article['published'].strftime('%d %B %Y %H:%M')}")

        # Extraire et afficher le contenu complet de l'article
        st.write("Chargement du contenu complet de l'article, et voici son résumé...")

        # Obtenir l'objet Article avec le contenu complet
        article_obj = get_article_content(article['link'])

        # Vérifier si l'article a déjà été traité (utilisation de processed_links)
        if article['link'] not in st.session_state.processed_links:
            # Générer un résumé pour l'article
            summary_content = generate_summary(article_obj.content)  # Passer le contenu de l'objet Article
            article_obj.summary = summary_content

            # Ajouter le lien de l'article à processed_links
            st.session_state.processed_links.append(article['link'])

            # Ajouter l'article traité à la liste des nouveaux articles
            stored_articles.append(article_obj)

            # Afficher le résumé
            st.write(summary_content)
        else:
            st.write("Cet article a déjà été traité.")
            # Afficher un résumé ou un message indiquant que l'article a déjà été traité
            stored_article = next((a for a in stored_articles if a.link == article['link']), None)
            if stored_article:
                # Assurez-vous que l'attribut summary existe dans l'article stocké
                st.write(getattr(stored_article, 'summary', 'Résumé non disponible'))

    # Sauvegarder les articles traités (ajoute les nouveaux articles au fichier JSON)
    save_processed_articles(stored_articles)

    st.markdown(f"[Lien vers l'article original]({article['link']})")
    st.markdown("---")  # Ligne de séparation entre les articles

    # Split the content into smaller chunks pour le traitement
    chunks = load_and_split_documents(stored_articles)
    
    if chunks:
        # Store embeddings for the chunks
        store_embeddings(chunks)  # Ajouter les embeddings à la VectorDB


# Récupérer les articles des dernières 24h
articles = get_rss_feed(RSS_URL)

# Afficher les articles dans Streamlit
display_articles(articles)

# Fonction pour interroger la base de données vectorielle FAISS
def query_faiss(vectordb, query_text):
    len_k = vectordb.index.ntotal
    k = max(1, min(int(len_k * 0.2), 80))

    print(f"len_k: {len_k}, Calculated k: {k}")  # Debug
    print(f"test : {len(vectordb.index_to_docstore_id)}")
    try:
        fetch_k = min(len_k, len(vectordb.index_to_docstore_id))
        results = vectordb.max_marginal_relevance_search(query_text, k=k, fetch_k=len_k, lambda_mult=0.7)
    except KeyError as e:
        print(f"KeyError: {e}")
        print("Vérifiez les indices dans index_to_docstore_id")
        print(f"Available keys: {vectordb.index_to_docstore_id.keys()}")
        raise

    # Convertir les résultats en une chaîne de caractères
    if results:
        results_str = "Résultats de la requête :\n"
        for result in results:
            results_str += f"- {result}\n"
    else:
        results_str = "Aucun résultat trouvé."

    template_rep=f"""
    Réponds à la question suivante ( {query_text} ) avec les informations ci-joint : {results_str}
    """

    response = client.chat.completions.create(
    messages=[{
        "role": "user",
        "content": template_rep,
    }],
    model="gpt-3.5-turbo",
    )

    return response.choices[0].message.content

# Exemple de recherche
search_query = st.text_input("Rechercher dans les actualités :")
# Bouton de recherche
if st.button("Rechercher"):
    if search_query:
        # if isinstance(search_query, bytes):
        #     search_query = search_query.decode('utf-8')
        results = query_faiss(st.session_state['vectordb'], search_query)
        if results:
            st.write(results)
    else:
        st.warning("Veuillez entrer une requête de recherche.")