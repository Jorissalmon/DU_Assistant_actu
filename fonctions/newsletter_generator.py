import os
import base64

# Fonction pour générer le contenu HTML
def generate_html_content(grouped_articles_by_topics, retrieval_time_format1, retrieval_time_format3):

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
        <em style="font-size: 0.6em;">{retrieval_time_format1}</em></h1>
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
        "Divers": "#545B5D",          # Gris sombre
        "Autres actus": "#000000"
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
    chemin="Tests_newsletter/24hActus_Newsletter_"+retrieval_time_format3+".html"
    # Enregistrer la newsletter dans un fichier HTML
    with open(chemin, "w", encoding="utf-8") as file:
        file.write(html_content)
    return html_content
