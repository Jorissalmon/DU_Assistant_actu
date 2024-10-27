import streamlit as st
import feedparser
import requests
import os
import json
import locale
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone
import openai
import faiss
import numpy as np
from openai import OpenAI
from typing import List
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores import FAISS
from langchain.schema import Document
from collections import defaultdict
from fuzzywuzzy import process

###################################
#########CONFIG OpenAI#############
###################################
OPENAI_API_KEY=st.secrets["OPENAI_API_KEY"]
st.session_state.OPENAI_API_KEY=OPENAI_API_KEY
client = OpenAI(
api_key=OPENAI_API_KEY,
)
st.session_state.client = client
# Configuration de l'API OpenAI
openai.api_key = OPENAI_API_KEY

###################################
#########Initialisation############
###################################
# URL du flux RSS
RSS_URL = "https://www.francetvinfo.fr/titres.rss"

# Fichier pour stocker les articles traités
PROCESSED_ARTICLES_FILE = "processed_articles.json"

# Définition de la classe Article
class Article:
    def __init__(self, title, link, content, chapo, published, summary=None, category=None, title_propre=None):
        """
        Initialise un article avec les informations de base, une catégorie par défaut à None.

        Args:
        - title (str): Le titre de l'article.
        - link (str): Le lien de l'article.
        - content (str): Le contenu de l'article.
        - chapo (str): Le résumé ou introduction de l'article.
        - published (str/datetime): La date de publication sous forme de chaîne ou d'objet datetime.
        - summary (str, optionnel): Un résumé de l'article.
        - category (str, optionnel): La catégorie de l'article, initialisée à None.
        - title_propre (str, optionnel): Un titre reformulé de l'article.
        """
        self.title = title
        self.link = link
        self.content = content
        self.chapo = chapo
        self.category = category
        self.title_propre = title_propre
        
        # Convertir published en datetime avec fuseau horaire si c'est une chaîne
        if isinstance(published, str):
            try:
                self.published = datetime.fromisoformat(published)
                if self.published.tzinfo is None:  # Si la date est naïve, ajouter un fuseau horaire (par exemple UTC)
                    self.published = self.published.replace(tzinfo=timezone.utc)
            except ValueError:
                print(f"Erreur: Format de date non valide pour l'article '{self.title}'")
                self.published = None  # ou définir une valeur par défaut si nécessaire
        elif isinstance(published, datetime):
            if published.tzinfo is None:  # Si la date est naïve, ajouter un fuseau horaire
                self.published = published.replace(tzinfo=timezone.utc)
            else:
                self.published = published
        else:
            print(f"Erreur: Type de date non pris en charge pour l'article '{self.title}'")
            self.published = None
        
        self.summary = summary
        self.metadata = {
            "title": self.title,
            "published": self.published.isoformat(),
            "title_propre": self.title_propre  # Inclure title_propre dans metadata
        }  

    def __str__(self):
        return (f"Title: {self.title}\n"
                f"Title Propre: {self.title_propre}\n"  # Afficher title_propre
                f"Link: {self.link}\n"
                f"Published: {self.published.strftime('%d %B %Y %H:%M')}\n"
                f"Chapo: {self.chapo}\n"
                f"Category: {self.category}\n"  # Ajout de l'affichage de la catégorie
                f"Content: {self.content[:100]}...")  # Affiche les 100 premiers caractères du contenu

    @property
    def page_content(self):
        return self.content  # Retourne le contenu de l'article comme page_content

    def to_dict(self):
        """
        Convertit l'objet Article en dictionnaire.

        Returns:
        - dict: Un dictionnaire avec les informations de l'article.
        """
        return {
            'title': self.title,
            'link': self.link,
            'chapo': self.chapo,
            'content': self.content,
            'published': self.published.isoformat(),
            'summary': self.summary,
            'category': self.category,
            'title_propre': self.title_propre,
            'metadata': self.metadata,
        }

    @staticmethod
    def from_dict(data):
        """
        Crée un objet Article à partir d'un dictionnaire.

        Args:
        - data (dict): Un dictionnaire avec les informations de l'article.

        Returns:
        - Article: Un objet Article.
        """
        published = datetime.fromisoformat(data['published'])
        return Article(
            title=data['title'], 
            link=data['link'], 
            chapo=data['chapo'], 
            content=data['content'], 
            published=published, 
            summary=data.get('summary', ""), 
            category=data.get('category', ""),
            title_propre=data.get('title_propre', "")
        )


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

###################################
#########DEF Fonctions#############
###################################
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

# Fonction pour récupérer le contenu d'un article
def get_article_content(link):
    response = requests.get(link)
    soup = BeautifulSoup(response.content, "html.parser")
    
    # Extraire le titre
    title_element = soup.find("h1", class_="c-title")
    title = title_element.get_text(strip=True) if title_element else ""
    
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

def generate_summary(article_content, title):
    client = st.session_state.client
    
    # 1. Préparer le message pour générer le résumé
    summary_message = {
        "role": "user",
        "content": f"""
        Résume cet article en mettant en avant les points clés de manière brieve et pédagogique. Assure-toi que l'utilisateur puisse comprendre l'actualité et les enjeux principaux. Tu ne dois pas dire "L'article raconte cela" mais plutot le résumer directement.
        Voici le contenu de l'article :\n\n{article_content} et {title}
        """
    }

    # 2. Envoyer la requête pour obtenir le résumé
    summary_response = client.chat.completions.create(
        messages=[summary_message],
        model="gpt-3.5-turbo",
    )

    # Extraire le résumé de la réponse
    summary_text = summary_response.choices[0].message.content.strip()
    
    # 3. Initialiser la variable pour le titre reformulé
    reformulated_title = None

    # 4. Si un titre est fourni, préparer le message pour la reformulation
    if title:
        reformulate_title_message = {
            "role": "user",
            "content": f"""
            Reformule le titre suivant '{title}', ajoute un emoji si besoin.
            """
        }

        # 5. Envoyer la requête pour obtenir le titre reformulé
        title_response = client.chat.completions.create(
            messages=[reformulate_title_message],
            model="gpt-3.5-turbo",
        )

        # Extraire le titre reformulé de la réponse
        reformulated_title = title_response.choices[0].message.content.strip()

    return summary_text, reformulated_title  # Retourner le résumé et le titre reformulé

# Fonction pour splitter les documents
def load_and_split_documents(documents, chunk_size=100, chunk_overlap=10):
    text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = text_splitter.split_documents(documents)
    docs_content = "\n\n".join([doc.page_content for doc in split_docs])
    # st.write(f"Nombre de chunks générés : {len(split_docs)}")  # Passer une liste d'objets Article
    return split_docs

# Fonction pour faire de l'embedding et vectoriser les docs
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

# Fonction pour résumer les articles, retourne tous les articles enregistrer depuis le départ
def articles_adds(articles):

    # Charger les articles déjà traités (retourne une liste complète d'articles)
    stored_articles = load_processed_articles()

    # Extraire les liens des articles déjà traités pour éviter les doublons
    if 'processed_links' not in st.session_state:
        st.session_state.processed_links = [article['link'] for article in stored_articles if 'link' in article]

    for article in articles:
        # st.header(article['title'])
        # st.write(f"*Publié le :* {article['published'].strftime('%d %B %Y %H:%M')}")

        # Obtenir l'objet Article avec le contenu complet
        article_obj = get_article_content(article['link'])

        # Vérifier si l'article a déjà été traité (utilisation de processed_links)
        if article['link'] not in st.session_state.processed_links:
            # Extraire et afficher le contenu complet de l'article
            # st.write("Chargement du contenu complet de l'article, et rédaction de son résumé...")

            # Générer un résumé pour l'article
        
            summary_content, reformulated_title = generate_summary(article_obj.content, title=article_obj.title)
            article_obj.summary = summary_content

            # Stocker le titre reformulé dans l'objet Article
            article_obj.title_propre = reformulated_title if reformulated_title else article['title'] 

            # Ajouter le lien de l'article à processed_links
            st.session_state.processed_links.append(article['link'])

            # Ajouter l'article traité à la liste des nouveaux articles
            stored_articles.append(article_obj)

            # Afficher le résumé
            # st.write(summary_content)
        # else:
            # st.write("Cet article a déjà été traité.")
            # Afficher un résumé ou un message indiquant que l'article a déjà été traité
            # stored_articles.append(next((a for a in stored_articles if a.link == article['link']), None))
            # if stored_article:
                # Assurez-vous que l'attribut summary existe dans l'article stocké
                # st.write(getattr(stored_article, 'summary', 'Résumé non disponible'))

    # Sauvegarder les articles traités (ajoute les nouveaux articles au fichier JSON)
    save_processed_articles(stored_articles)

    # st.markdown(f"[Lien vers l'article original]({article.link})")
    st.markdown("---")  # Ligne de séparation entre les articles

    # Split the content into smaller chunks pour le traitement
    chunks = load_and_split_documents(stored_articles)
    
    if chunks:
        # Store embeddings for the chunks
        store_embeddings(chunks)  # Ajouter les embeddings à la VectorDB
    return stored_articles

# Fonction pour interroger ChatGPT et obtenir la classification des titres
def classify_article_titles_with_gpt(articles_titres):
    # Créer une liste de tous les titres
    titles = [article.title for article in articles_titres]
    
    # Créer la requête pour GPT avec un prompt amélioré
    prompt = (
        "Voici une liste de titres d'articles d'actualité. "
        "Votre tâche est de classer chacun de ces titres dans une catégorie appropriée. Vous devez tous les classer"
        "Supprime les Titre introuvable"
        "Les catégories incluent : 'Politique', 'Technologie', 'International', 'Économie', 'Culture', 'Environnement', 'Santé', 'Sport', 'Justice', 'Faits divers', 'Autres actus', et 'Sociétés'."
        "Pour chaque titre, indiquez la catégorie correspondante en utilisant le format 'Titre | Catégorie'. "
        "Voici des exemples de titres pour référence :\n"
        "- Titre : Apologie du terrorisme : des plaques de rues recouvertes de noms de leaders du Hamas à Poitiers | Politique\n"
        "- Titre : La garantie individuelle du pouvoir d'achat ne sera pas versée aux fonctionnaires cette année | Économie\n"
        "Voici les titres à classifier :\n" + "\n".join(titles)
    )
    
    # Envoyer la requête à GPT
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{
            "role": "user",
            "content": prompt,
        }]
    )
    
    # Extraire la réponse
    classification_text = response.choices[0].message.content
    # print(f"Texte de classification brut : {classification_text}")  # Debug

    # Transformer la réponse en un dictionnaire
    classified_titles = classification_text.split("\n")
    classification = {}

    for line in classified_titles:
        line = line.strip()  # Supprimer les espaces superflus
        if "|" in line:
            parts = line.split("|", 1)  # Limiter à 1 pour éviter le problème de plusieurs ": "
            if len(parts) == 2:  # Vérifier qu'on a bien deux parties
                title = parts[0].strip()
                category = parts[1].strip()
                # Exclure les titres introuvables
                if title.lower() != "Titre introuvable":
                    classification[title] = category
            else:
                print(f"Ligne ignorée, format inattendu : {line}")  # Alerte sur le format inattendu
        else:
            print(f"Ligne ignorée, format inattendu : {line}")  # Alerte sur le format inattendu
    print(classification)
    return classification

# Fonction pour classifier et regrouper les articles par catégorie
def group_articles_by_topic(articles):
    """
    Regroupe les articles par catégories après avoir classifié les titres via GPT.

    Args:
    - articles: Une liste d'objets d'articles contenant les titres.

    Returns:
    - grouped_articles: Un dictionnaire contenant les articles regroupés par catégorie.
    """

    # Obtenir la classification des titres via GPT
    classification = classify_article_titles_with_gpt(articles)

    # Créer un dictionnaire pour stocker les articles regroupés par catégorie
    grouped_articles = defaultdict(list)

    # Liste des titres classifiés
    classified_titles = list(classification.keys())

    # Seuil de correspondance pour accepter une classification
    threshold = 85

    # Assigner chaque article à la catégorie correspondante
    for article in articles:
        # Correspondance approximative entre le titre de l'article et la classification des titres
        best_match, score = process.extractOne(article.title, classified_titles)

        if score >= threshold:
            # Obtenir la catégorie correspondante depuis la classification
            category = classification[best_match]
            article.category = category  # Mettre à jour l'article avec la catégorie correspondante
        else:
            # Si le score est inférieur au seuil, on attribue la catégorie "Divers"
            category = "Autres actus"
            article.category = category

        # Ajouter l'article au dictionnaire de regroupement
        grouped_articles[category].append(article)

    return grouped_articles


###################################
#########APPLICATION###############
#########Articles 24h##############
###################################
st.title("Actualités des dernières 24H 🌐")
st.write("**Source** : [France Info](https://www.francetvinfo.fr/)")

# Ajouter un bouton pour charger l'actualité des dernières 24 heures
if st.button("Charger l'actualité des dernières 24 heures"):

    # Récupérer les articles des dernières 24h
    articles_titres = get_rss_feed(RSS_URL)

    # Tous les articles qui sont enregistrés
    articles = articles_adds(articles_titres)

    # Calculer la limite des dernières 24 heures
    last_24h = datetime.now(timezone.utc) - timedelta(hours=24)

    # Filtrer les articles publiés dans les dernières 24 heures
    articles_24h = [article for article in articles if article.published and article.published > last_24h]

    # Trier les articles par date décroissante
    articles_24h_sorted = sorted(articles_24h, key=lambda x: x.published, reverse=True)

    grouped_articles_by_topics = group_articles_by_topic(articles_24h_sorted)
    st.session_state.grouped_articles_by_topics = grouped_articles_by_topics

    # Stocker la date et l'heure de récupération dans la session
    retrieval_time = datetime.now(timezone.utc)
    st.session_state.retrieval_time = retrieval_time.strftime("%A %d %B %Y à %H:%M")


# Afficher la date et l'heure de récupération à côté du bouton
if 'retrieval_time' in st.session_state:
    st.write(f"Données récupérées le : **{st.session_state.retrieval_time}**")

###################################
#########APPLICATION###############
#########Interroger Actu###########
###################################
# Fonction pour interroger la base de données vectorielle FAISS
def query_faiss(vectordb, query_text):
    len_k = vectordb.index.ntotal
    k = max(1, min(int(len_k * 0.2), 80))

    # print(f"len_k: {len_k}, Calculated k: {k}")  # Debug
    # print(f"test : {len(vectordb.index_to_docstore_id)}")
    try:
        fetch_k = min(len_k, len(vectordb.index_to_docstore_id))
        results = vectordb.max_marginal_relevance_search(query_text, k=k, fetch_k=len_k, lambda_mult=0.7)
    except KeyError as e:
        print(f"KeyError: {e}")
        print("Vérifiez les indices dans index_to_docstore_id")
        # print(f"Available keys: {vectordb.index_to_docstore_id.keys()}")
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
    Réponds avec le maximum d'information et d'un point de vue pédagogique
    """

    response = client.chat.completions.create(
    messages=[{
        "role": "user",
        "content": template_rep,
    }],
    model="gpt-3.5-turbo",
    )

    return response.choices[0].message.content

st.sidebar.write("💬 Vous pouvez également poser une question sur l'actualité !")
search_query = st.sidebar.text_input("Rechercher dans les actualités :")
# Bouton de recherche
if st.sidebar.button("Rechercher"):
    if search_query:
        # if isinstance(search_query, bytes):
        #     search_query = search_query.decode('utf-8')
        results = query_faiss(st.session_state['vectordb'], search_query)
        if results:
            st.write(results)
    else:
        st.warning("Veuillez entrer une requête de recherche.")

###################################
#########APPLICATION###############
#########Newsletter Actu###########
###################################
import base64
import streamlit.components.v1 as components

# Fonction pour générer le contenu HTML
def generate_html_content(grouped_articles_by_topics):
    # Créer une structure HTML pour la newsletter
    current_date = datetime.now()

    # Fonction pour convertir une image en base64
    def image_to_base64(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    # Convertir l'image en base64
    background_image_base64 = image_to_base64("background_newsletter.png")

    def hex_to_rgba(hex_color, alpha=0.8):
        hex_color = hex_color.lstrip("#")
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f"{r}, {g}, {b}, {alpha}"

    html_content = f"""
    <html>
        <head>
            <meta charset="UTF-8">
            <title>Newsletter</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background: #f4f4f4; /* Couleur de fond de la page */
                background-attachment: fixed;
                background-image: url(data:image/png;base64,{background_image_base64});
                background-size: cover; /* Recouvrir tout l'écran */
                backdrop-filter: blur(10px); /* Flou de l'arrière-plan */
                color: #333;
                text-align: center; /* Centrer le texte dans le corps */
            }}
            h1 {{
                color: white;                               /* Couleur du texte en blanc */
                font-weight: bold;                          /* Texte en gras */
                background-color: rgba(0, 0, 0, 0.7);       /* Fond noir avec transparence */
                padding: 20px;                              /* Espacement interne */
                border-radius: 10px;                         /* Coins arrondis */
                backdrop-filter: blur(10px);                /* Effet de flou sur l'arrière-plan */
                border: 10px solid white;                    /* Bordure blanche autour du titre */
                width: 100%;                                /* Prendre toute la largeur */
                text-align: center;                         /* Centrer le texte */
                margin: 0;                                  /* Supprimer les marges par défaut */
                position: relative;                         /* Positionner correctement l'élément */
                box-sizing: border-box;                     /* Inclure la bordure dans la largeur */
            }}
            .category-title {{
                display: inline-block;
                width: 200px;
                padding: 10px 20px; /* Espacement autour du texte */
                background-color: rgba(255, 255, 255, 1); /* Fond blanc transparent */
                color: {{category_text_color}}; /* Couleur du texte selon la catégorie */
                border-radius: 10px 10px 0 0;   /* Coins arrondis seulement en haut */
                margin: 20px 0px;                 /* Espace au-dessus et au-dessous */
                font-weight: bold;
                text-align: center;               /* Aligner */
            }}
            ul {{
                list-style-type: none;
                backdrop-filter: blur(10px); /* Flou de l'arrière-plan */
            }}
            li {{
                margin: 20px auto; /* Centre les articles */
                padding: 15px;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                max-width: 800px; /* Largeur maximale pour centrer */
                text-align: justify;
                color: white; /* Texte en blanc pour les articles */
                background-color: rgba(255, 255, 255, 0.8); /* Fond blanc avec transparence */
            }}
            strong {{
                color: white; /* Le texte fort en blanc */
            }}
        </style>
    </head>
    <body>
        <h1>📰 Newsletter<br>
        <em style="font-size: 0.6em;">{st.session_state.retrieval_time}</em></h1>
    """

    # Couleurs associées aux catégories (version plus sombre)
    category_colors = {
        "Politique": "#6A98D9",       # Bleu sombre
        "Technologie": "#8A8D90",     # Gris sombre
        "International": "#DFAF5D",   # Orange sombre     
        "Économie": "#77C4DB",        # Cyan sombre
        "Culture": "#A478C4",         # Violet sombre
        "Environnement": "#6ABBA5",   # Vert sombre
        "Santé": "#4CB8B1",           # Vert menthe sombre
        "Sport": "#E49A9E",           # Rouge sombre
        "Justice": "#E6C94A",         # Jaune sombre
        "Faits divers": "#E17677",    # Rouge clair sombre
        "Société": "#BFA0C5",         # Violet doux sombre
        "Divers": "#545B5D"           # Gris sombre
    }

    # Ordre de priorité des catégories par importance
    priority_order = [
        "Politique",
        "Économie",
        "Technologie",
        "Environnement",
        "Santé",
        "International",
        "Justice",
        "Société",
        "Culture",
        "Sport",
        "Faits divers",
        "Autres actus"
    ]

    # Filtrer les catégories selon l'ordre de priorité défini et ne garder que celles qui contiennent des articles
    ordered_categories = [category for category in priority_order if category in grouped_articles_by_topics and grouped_articles_by_topics[category]]


    # Ajouter chaque catégorie et ses articles à la newsletter
    for category in ordered_categories:
        articles = grouped_articles_by_topics[category]

        # Si la catégorie contient des articles
        if len(articles)>0:
            # Obtenir la couleur de fond et du texte pour la catégorie
            article_background_color = category_colors.get(category, "#808080")
            category_text_color = category_colors.get(category, "#000000")  # Texte en noir si non défini

            # Ajouter le titre de la catégorie avec sa couleur
            html_content += f"<h2 class='category-title' style='color: {category_text_color};'>{category}</h2><ul>"

            # Ajouter chaque article de la catégorie
            for article in articles:
                if article.title != "":
                    html_content += f"""
                    <li style='background-color: rgba({hex_to_rgba(category_colors[category], 0.85)});'>
                        <strong><em>{article.title_propre}</em></strong><br><br>
                        {article.summary}
                    </li>
                    """

            html_content += "</ul>"

    html_content += """
        </body>
    </html>
    """

    st.session_state.newsletter=html_content

    # Enregistrer la newsletter dans un fichier HTML
    with open("newsletter.html", "w", encoding="utf-8") as file:
        file.write(html_content)
    return html_content

col1, col2 = st.columns(2)

with col1:
# Générer et afficher la newsletter
    if st.button("Générer la newsletter"):
        # Générer le contenu HTML de la newsletter
        html_content_generation = generate_html_content(st.session_state.grouped_articles_by_topics)

        # Bouton de téléchargement de la newsletter
        if 'newsletter' in st.session_state:
            st.sidebar.download_button(
                label="Télécharger la Newsletter du jour",
                data=html_content_generation,
                file_name="Newsletter_du_" + st.session_state.retrieval_time + ".html", 
                mime="text/html"
            )

if 'newsletter' in st.session_state:
    # Lire le contenu du fichier HTML
    with open("newsletter.html", "r", encoding="utf-8") as file:
        newsletter_html = file.read()
    # Afficher la newsletter dans Streamlit
    components.html(newsletter_html, height=800, width=800, scrolling=True)

###################################
#########APPLICATION###############
#########PODCAST Actu##############
###################################
import os
from pydub import AudioSegment
# Créer un dossier pour stocker les fichiers audio individuels
def create_speech_directory():
    if not os.path.exists("speech"):
        os.makedirs("speech")

def generate_podcast_script(articles_by_category):
    podcast_script = []
    
    # Introduction
    podcast_script.append({
        "voice": "masculine", 
        "text": f"""
        Salut, et bienvenue dans ton podcast 24 heures Actu présentant les actualités des dernières 24 heures à partir du {st.session_state.retrieval_time} !
        """
    })
    
    podcast_script.append({
        "voice": "feminine", 
        "text": """
        Installes toi confortablement et prépare toi à tout connaitre des infos récentes du moment !
        """
    })
    
    # Alterner entre les catégories et les articles
    alternating_voice = "masculine"  # Commencer par la voix masculine

    for category, articles in articles_by_category.items():
        if category!="Autres actus":
            # Présenter chaque catégorie
            if alternating_voice == "masculine":
                podcast_script.append({
                    "voice": "masculine",
                    "text": f"Allons-y, parlons des actus {category} !"
                })
                alternating_voice = "feminine"
            else:
                podcast_script.append({
                    "voice": "feminine",
                    "text": f"Maintenant, découvrons ensemble ce qui se passe dans les actus {category} !"
                })
                alternating_voice = "masculine"

            # Ajouter les articles
            for article in articles:
                if alternating_voice == "masculine":
                    podcast_script.append({
                        "voice": "masculine",
                        "text": f"{article.title}. {article.summary}"
                    })
                    alternating_voice = "feminine"
                else:
                    podcast_script.append({
                        "voice": "feminine",
                        "text": f"{article.title}. {article.summary}"
                    })
                    alternating_voice = "masculine"

    # Conclusion
    podcast_script.append({
        "voice": "feminine",
        "text": """
        Et voilà, c'est la fin de notre épisode sur les actus des dernières 24 heures.
        Merci à toi d'avoir écouter cette épisode, retrouves ces actualités dans un formats newsletter sur notre site
        """
    })

    podcast_script.append({
        "voice": "masculine",
        "text": """
        N'oublies pas de revenir demain pour encore plus d'actus.
        Prends soin de toi et à demain !
        """
    })
    
    return podcast_script

def text_to_mp3_Openai(text, voice, output_path="audio.mp3"):
    # Utilisation de l'API OpenAI pour la synthèse vocale
    client = st.session_state.client
    response = client.audio.speech.create(
        model="tts-1",
        voice=voice,  # Voix dynamique basée sur le sexe
        input=text,
    )
    response.stream_to_file(output_path)
    return output_path

# Fonction pour générer le podcast entier
def generate_podcast_audio(podcast_script):
    create_speech_directory()  # Créer le dossier speech s'il n'existe pas
    combined_audio = AudioSegment.silent(duration=1000)  # Commencer avec 1s de silence
    
    for i, segment in enumerate(podcast_script):
        voice = "echo" if segment["voice"] == "masculine" else "shimmer"  # Choisir la voix masculine/féminine
        output_path = f"speech/segment_{i+1}.mp3"  # Nom du fichier pour chaque segment
        
        # Générer et sauvegarder chaque segment audio
        text_to_mp3_Openai(segment["text"], voice, output_path)
        
        # Charger l'audio généré et le combiner
        audio_segment = AudioSegment.from_mp3(output_path)
        combined_audio += audio_segment  # Ajouter chaque segment d'audio

    # Chemin du fichier de sortie pour l'audio combiné
    combined_output_path = "speech/podcast_final.mp3"
    
    # Exporter l'audio combiné en un seul fichier MP3
    combined_audio.export(combined_output_path, format="mp3")
    
    return combined_output_path

with col2:
    # Générer le script du podcast
    if st.button("Générer le podcast"):
        podcast_script = generate_podcast_script(st.session_state.grouped_articles_by_topics)
        st.session_state.podcast_script=podcast_script

        if st.sidebar.button("Générer le podcast audio"):
            # Générer l'audio
            podcast_audio = generate_podcast_audio(podcast_script)
            # Proposer le téléchargement du podcast
            st.sidebar.download_button(
                label="Télécharger le podcast en MP3",
                data=podcast_audio,
                file_name="podcast.mp3",
                mime="audio/mpeg"
            )
if 'podcast_script' in st.session_state:
    st.write(st.session_state.podcast_script)

###################################
#########APPLICATION###############
#########Envoie Mail Actu##########
###################################
# import re

# def is_valid_email(email):
#     # Utilisation d'une regex pour vérifier si l'email est au bon format
#     regex = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
#     return re.match(regex, email) is not None

# import os.path
# import base64
# from email.message import EmailMessage
# import google.auth
# from googleapiclient.discovery import build
# from googleapiclient.errors import HttpError
# from google.oauth2.credentials import Credentials
# from google_auth_oauthlib.flow import InstalledAppFlow
# from google.auth.transport.requests import Request

# # Si vous modifiez ces portées, supprimez le fichier token.json
# SCOPES = ['https://www.googleapis.com/auth/gmail.compose']

# def gmail_authenticate():
#     """Authentification OAuth2 pour Gmail API."""
#     creds = None
#     # Le fichier token.json stocke les tokens d'accès et de rafraîchissement de l'utilisateur
#     if os.path.exists('token.json'):
#         creds = Credentials.from_authorized_user_file('token.json', SCOPES)
#     # Si il n'y a pas de token valide, il faut s'authentifier
#     if not creds or not creds.valid:
#         if creds and creds.expired and creds.refresh_token:
#             creds.refresh(Request())
#         else:
#             flow = InstalledAppFlow.from_client_secrets_file(
#                 'credentials.json', SCOPES)
#             creds = flow.run_local_server(port=0)
#         # Sauvegarder le token pour la prochaine exécution
#         with open('token.json', 'w') as token:
#             token.write(creds.to_json())
#     return creds

# def gmail_create_draft(sender_email, recipient_email, subject, newsletter_content):
#     """Créer un brouillon dans Gmail."""
#     creds = gmail_authenticate()

#     try:
#         # Créer le client Gmail API
#         service = build("gmail", "v1", credentials=creds)

#         message = EmailMessage()

#         # Contenu du mail
#         message.set_content(newsletter_content)

#         message["To"] = recipient_email
#         message["From"] = sender_email
#         message["Subject"] = subject

#         # Encoder le message au format base64
#         encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()

#         create_message = {"message": {"raw": encoded_message}}

#         # Créer le brouillon dans Gmail
#         draft = (
#             service.users()
#             .drafts()
#             .create(userId="me", body=create_message)
#             .execute()
#         )

#         # print(f'Draft created with id: {draft["id"]}\nDraft message: {draft["message"]}')
#         return draft

#     except HttpError as error:
#         print(f"An error occurred: {error}")
#         return None

# recipient_email = st.text_input("Entrez l'adresse e-mail pour recevoir la newsletter :")

# Définir les informations du mail
# sender_email = "joris.entrepro@gmail.com"
# recipient_email = recipient_email
# subject = "Votre Newsletter"
# newsletter_content = """
# Bonjour,

# Voici les dernières actualités de notre newsletter !

# Cordialement,
# L'équipe.
# """

# if st.button("Envoyer la newsletter"):
#     if recipient_email:
#         if is_valid_email(recipient_email):
#             newsletter_text = generate_newsletter(articles)
#             # Créer un brouillon d'email
#             gmail_create_draft(sender_email, recipient_email, subject, newsletter_content)
#         else:
#             st.error("L'adresse e-mail est invalide. Veuillez entrer une adresse e-mail valide.")
#     else:
#         st.warning("Veuillez entrer une adresse e-mail.")



#################################################### Partage sur YOUTUBE

# from moviepy.editor import AudioFileClip, ImageClip, CompositeVideoClip

# def generate_final_video(audio_file_path, image_path, output_video_path):
#     # Charger l'audio
#     audio_clip = AudioFileClip(audio_file_path)
    
#     # Charger une image comme fond
#     image_clip = ImageClip(image_path)
#     image_clip = image_clip.set_duration(audio_clip.duration)
#     image_clip = image_clip.set_fps(24)  # 24 images par seconde pour la vidéo

#     # Créer un clip vidéo composite
#     video_clip = CompositeVideoClip([image_clip.set_audio(audio_clip)])

#     # Écrire la vidéo dans le fichier
#     video_clip.write_videofile(output_video_path, codec='libx264', audio_codec='aac')

#     # Libérer les ressources
#     audio_clip.close()
#     image_clip.close()
#     video_clip.close()

# # Exemple d'utilisation
# audio_file_path = './speech/podcast.mp3'  # Chemin vers le fichier audio
# image_path = './assets/thumbnail.jpg'  # Chemin vers une image à utiliser comme fond
# output_video_path = './/podcast_video.mp4'  # Chemin pour le fichier vidéo final

# generate_final_video(audio_file_path, image_path, output_video_path)