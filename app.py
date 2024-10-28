import streamlit as st
import feedparser
import requests
import os
import json
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone
import openai
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
            "published": self.published.isoformat()
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
                        chapo=article['chapo'], 
                        content=article['content'], 
                        published=article['published'], 
                        summary=article['summary'], 
                        category=article['category'],
                        title_propre=article['title_propre']
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

#Génère un résumé de l'article
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
        
            summary_content, reformulated_title = generate_summary(article_obj.content, article_obj.title)
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
        "Les catégories incluent : 'Politique', 'Technologie', 'International', 'Économie', 'Culture', 'Environnement', 'Santé', 'Sport', 'Justice', 'Faits divers', 'Autres actus', et 'Société'."
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
    st.session_state.retrieval_time_format1 = retrieval_time.strftime("%A %d %B %Y à %H:%M")
    st.session_state.retrieval_time_format2 = retrieval_time.strftime("%A %d %B %Y à %H heures %M")
    st.session_state.retrieval_time_format3 = retrieval_time.strftime("%A_%d_%B_%Y")

# Afficher la date et l'heure de récupération à côté du bouton
if 'retrieval_time1' in st.session_state:
    st.write(f"Données récupérées le : **{st.session_state.retrieval_time_format1}**")

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

st.sidebar.write("💬 **Vous pouvez également poser une question sur l'actualité !**")
search_query = st.sidebar.text_input("Rechercher dans les actualités :")
# Bouton de recherche
if st.sidebar.button("Rechercher"):
    if search_query:
        # if isinstance(search_query, bytes):
        #     search_query = search_query.decode('utf-8')
        results = query_faiss(st.session_state['vectordb'], search_query)
        if results:
            st.sidebar.write(results)
    else:
        st.warning("Veuillez entrer une requête de recherche.")

###################################
#########APPLICATION###############
#########Newsletter Actu###########
###################################
import base64
import streamlit.components.v1 as components
from fonctions.newsletter_generator import generate_html_content

col1, col2 = st.columns(2)

with col1:
# Générer et afficher la newsletter
    if st.button("Générer la newsletter"):

        # Générer le contenu HTML de la newsletter
        html_content_generation = generate_html_content(st.session_state.grouped_articles_by_topics, st.session_state.retrieval_time_format1, st.session_state.retrieval_time_format3)
        st.session_state.newsletter=html_content_generation

# Bouton de téléchargement de la newsletter
if 'newsletter' in st.session_state:
    st.sidebar.download_button(
        label="Télécharger la Newsletter du jour",
        data=st.session_state.newsletter,
        file_name="Newsletter_du_" + st.session_state.retrieval_time_format3 + ".html", 
        mime="text/html"
    )

if 'newsletter' in st.session_state:
    # Afficher la newsletter dans Streamlit
    components.html(st.session_state.newsletter, height=800, width=800, scrolling=True)

###################################
#########APPLICATION###############
#########PODCAST Actu##############
###################################
from fonctions.podcast_generator import generate_podcast_script, generate_podcast_audio_files, generate_podcast_audio_with_music, generate_final_video
from fonctions.upload_youtube import upload_video

with col2:
    # Générer le script du podcast
    if st.button("Générer le podcast"):
        audio_file_path = "Production_audio_visuel/24hActus_podcast_"+st.session_state.retrieval_time_format3+".mp3"
        image_path = 'miniature.png'
        output_video_path = "Production_audio_visuel/24hActus_Podcast" + st.session_state.retrieval_time_format3 + ".mp4"

        # Générer le podcast en prenant seulement les deux premières catégories
        # limited_articles = dict(list(st.session_state.grouped_articles_by_topics.items())[:2])  # Prendre les deux premières catégories

        # Générer le script du podcast
        podcast_script = generate_podcast_script(st.session_state.grouped_articles_by_topics, st.session_state.retrieval_time_format3)

        # Créer un fichier .txt pour sauvegarder le script
        txt_file_path = "podcast_script.txt"

        # Ouvrir le fichier en mode écriture et sauvegarder chaque segment
        with open(txt_file_path, "w", encoding="utf-8") as file:
            for segment in podcast_script:
                file.write(f"Voice: {segment['voice']}\n")
                file.write(f"Text: {segment['text']}\n")
                file.write("\n")  # Ajouter une ligne vide entre chaque segment

        # with open("podcast_script.txt", "r", encoding="utf-8") as file:
        #     podcast_script_content = file.read()  # Lire tout le contenu du fichier

        st.session_state.podcast_script=podcast_script

        # Générer les fichiers audio individuels Text to Speech
        generate_podcast_audio_files(st.session_state.podcast_script)

        # Ajout de la musique et des effets au podcast
        podcast_audio = generate_podcast_audio_with_music(podcast_script, "music.mp3")
        st.session_state.podcast_audio=podcast_audio
        st.sidebar.audio(podcast_audio) #Affichage sur le streamlit

        # Générer le fichier mp4 avec l'audio et la miniature
        generate_final_video(audio_file_path, image_path, output_video_path)
        
        # Uploader la video sur Youtube
        if st.sidebar.button("Upload sur Youtube"):
            file_name="24hActus_Podcast" + st.session_state.retrieval_time_format3 + ".mp4"
            description = f"""
            Ici, on te présente les actus du {st.session_state.retrieval_time_format3}
            N'hésites pas à nous faire un retour en commentaire
            """
            upload_video(file_path="podcast_video.mp4",title=file_name,description=description)

if 'podcast_audio' in st.session_state:
    # Proposer le téléchargement du podcast
    file_name="24hActus_Podcast" + st.session_state.retrieval_time_format3 + ".mp3"
    st.sidebar.download_button(
        label="Télécharger le podcast en MP3",
        data=st.session_state.podcast_audio,
        file_name=file_name,
        mime="audio/mpeg"
    )

if 'podcast_script' in st.session_state:
    st.write(st.session_state.podcast_script)

###################################
#########APPLICATION###############
#########Envoie Mail Actu##########
###################################

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