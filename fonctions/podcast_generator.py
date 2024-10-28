import os
import numpy as np
from pydub import AudioSegment
from pydub.effects import speedup
from openai import OpenAI
from scipy.signal import resample
from moviepy.editor import AudioFileClip, ImageClip, CompositeVideoClip
from fonctions.upload_youtube import upload_video

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

# Créer un dossier pour stocker les fichiers audio individuels
def create_speech_directory():
    if not os.path.exists("speech"):
        os.makedirs("speech")

# Génère le script du podcast avec les différents intervenants
def generate_podcast_script(articles_by_category, retrieval_time_format2):
    podcast_script = []

    # Introduction
    podcast_script.append({
        "voice": "masculine", 
        "text": f"""
        Salut, et bienvenue dans ton podcast 24 heures Actu présentant les actualités des dernières 24 heures à partir du {retrieval_time_format2} !
        """
    })
    
    podcast_script.append({
        "voice": "feminine", 
        "text": """
        Installes toi confortablement et prépare toi à tout connaitre des infos récentes du moment !
        """
    })
    
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
    ordered_categories = [category for category in priority_order if category in articles_by_category and articles_by_category[category]]


    # Alterner entre les catégories et les articles
    alternating_voice = "masculine"  # Commencer par la voix masculine

    for category in ordered_categories:
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
            for article in articles_by_category[category]:
                if alternating_voice == "masculine":
                    podcast_script.append({
                        "voice": "masculine",
                        "text": f"{article.title_propre}. {article.summary}"
                    })
                    alternating_voice = "feminine"
                else:
                    podcast_script.append({
                        "voice": "feminine",
                        "text": f"{article.title_propre}. {article.summary}"
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
        N'oublies pas de revenir demain pour encore plus d'actu.
        Prends soin de toi et à demain !
        """
    })
    
    return podcast_script

# Requete à l'API OpenAI pour les voix
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
def generate_podcast_audio_files(podcast_script):
    create_speech_directory()  # Créer le dossier speech s'il n'existe pas
    
    for i, segment in enumerate(podcast_script):
        voice_type = "echo" if segment["voice"] == "masculine" else "shimmer"  # Sélectionner la voix
        output_path = f"speech/segment_{i+1}.mp3"  # Sauvegarde de chaque segment

        # Générer chaque segment audio à partir du texte (via OpenAI ou un autre service)
        text_to_mp3_Openai(segment["text"], voice_type, output_path)
    
    return "Tous les fichiers audio ont été générés."

# Modification des voix
def change_voice_pitch(audio_segment, semitone_change):
    # Convertir le fichier AudioSegment en numpy array
    samples = np.array(audio_segment.get_array_of_samples())
    
    # Taux d'échantillonnage
    sample_rate = audio_segment.frame_rate
    
    # Calculer le facteur de changement de fréquence
    pitch_factor = 2 ** (semitone_change / 12.0)
    
    # Modifier le pitch en changeant la fréquence d'échantillonnage
    new_sample_rate = int(sample_rate * pitch_factor)
    
    # Resampling pour changer le pitch
    shifted_samples = resample(samples, int(len(samples) * (new_sample_rate / sample_rate)))
    
    # Convertir les samples retournés en AudioSegment
    return audio_segment._spawn(shifted_samples.astype(audio_segment.array_type).tobytes())

#Ajoutes les effets 
def generate_podcast_audio_with_music(podcast_script, background_music_path, retrieval_time_format3):
    # Créer le dossier speech s'il n'existe pas
    create_speech_directory()

    combined_audio = AudioSegment.silent(duration=500)  # 1 seconde de silence au début
    
    # Charger la musique de fond
    background_music = AudioSegment.from_file(background_music_path)
    
    # Variables pour ajuster les voix et la musique
    voice_volume = +10  # Augmenter un peu le volume de la voix
    music_volume = -10  # Baisser le volume de la musique de fond pour ne pas dominer la voix
    intro_duration = 15000  # Durée de la musique d'intro en millisecondes (15 secondes)
    outro_duration = 15000  # Durée de la musique d'outro en millisecondes (15 secondes)

    for i, segment in enumerate(podcast_script):
        voice_type = "echo" if segment["voice"] == "masculine" else "shimmer"  # Sélectionner la voix
        output_path = f"speech/segment_{i+1}.mp3"  # Chemin du fichier MP3

        # Charger le segment audio généré
        voice_audio = AudioSegment.from_mp3(output_path)
        voice_audio = voice_audio + voice_volume  # Ajuster le volume de la voix
        
        # Ajuster la hauteur de la voix
        # if voice_type == "echo":  # Voix masculine
        #     voice_audio = change_voice_pitch(voice_audio, semitone_change=-0.5)  # Plus grave
        # elif voice_type == "shimmer":  # Voix féminine
        #     voice_audio = change_voice_pitch(voice_audio, semitone_change=0.5)  # Plus aiguë

        # Accélérer la voix de 1.25x
        voice_audio = speedup(voice_audio, playback_speed=1.25)

        # Ajouter musique à l'introduction
        if i == 0:
            background_music_intro = background_music[:intro_duration].fade_in(2000).fade_out(2000) + music_volume
            combined_segment = background_music_intro.overlay(voice_audio)
        # Ajouter musique à la conclusion dans les 15 dernières secondes
        elif i == len(podcast_script) - 2:
            # Avant-dernière voix: commence la musique de fond ici
            outro_length = len(voice_audio)  # Longueur de l'avant-dernière voix
            background_music_outro = background_music[:outro_length].fade_in(outro_length) + music_volume  # Prend la musique jusqu'à la longueur de la voix
            combined_segment = voice_audio.overlay(background_music_outro)

        elif i == len(podcast_script) - 1:
            # Dernière voix: reprendre la musique après la voix
            outro_voice_part = voice_audio  # Prendre la voix entière

            # Reprendre la musique où elle s'est arrêtée dans l'avant-dernière voix
            start_position = len(podcast_script[-2])  # Longueur de l'avant-dernière voix pour savoir où reprendre la musique
            background_music_resume = background_music[9800:23000].fade_out(3000) + music_volume

            # Combiner la dernière voix avec la musique de fond
            combined_segment = background_music_resume.overlay(outro_voice_part)

        else:
            # Pas de musique pour les segments intermédiaires
            combined_segment = voice_audio
                
        # Ajouter le segment combiné à l'audio final
        combined_audio += combined_segment

    # Chemin du fichier de sortie pour l'audio combiné
    combined_output_path = "Production_audio_visuel/24hActus_podcast_"+retrieval_time_format3+".mp3"
    
    # Exporter l'audio combiné en un seul fichier MP3
    combined_audio.export(combined_output_path, format="mp3")
    
    return combined_output_path

#Générer la conversion en video
def generate_final_video(audio_file_path, image_path, output_video_path):
    # Charger le fichier audio
    audio_clip = AudioFileClip(audio_file_path)
    
    # Charger l'image
    image_clip = ImageClip(image_path)
    
    # Définir la durée de l'image à la durée de l'audio
    image_clip = image_clip.set_duration(audio_clip.duration)
    
    # Ajouter l'audio à l'image
    video_clip = image_clip.set_audio(audio_clip)

    # Écrire le résultat final dans un fichier
    video_clip.write_videofile(output_video_path,fps=1)
