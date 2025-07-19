import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from fpdf import FPDF
import tempfile
import warnings
warnings.filterwarnings('ignore')
# configration de la page
st.set_page_config(page_title='Mon Assistant Santé : Évaluation de risque de cancer de sein',
                   page_icon='🩺',
                   layout='wide',
                   initial_sidebar_state='expanded')

@st.cache_resource
def load_composents():
    try:
        model = joblib.load('logistique.pkl')
        scaler = joblib.load('scaler.pkl')
        features = joblib.load('features.pkl')
        return model,scaler,features
    except FileNotFoundError as e:
        st.error(f'Oups ! il semble qu\'un fichier soit manquant :{e}')
        st.info('Assurez-vous que tous les fichiers du modele sont présent dans le dossier')
        st.stop()
    except Exception as e:
        st.error(f'Une erreur c\'est produite lors de chargement :{e}')
        st.info('Veuillez vérifier l\'ntégrité des fichiers du moele')
        st.stop()
model,scaler,features = load_composents()
st.markdown('''
 <style>
 .main-header{
    background: linear-gradient(135deg,#4CAF50 0%,#2196F3 100%);
    padding:2.5rem;
    border-radius:30px;  
    margin-bottom:2rem;
    box-shadow:0 10px 40px rgba(0,0,0,0.1);
    color:white;
    text-align:center;                      
     }
 </style>           
''',unsafe_allow_html=True)
# en tete
st.markdown('''
            <div class='main-header'>
            <h1>🩺 Mon Assistant Santé</h1>
            <h3>Votre compagnon pour l'évaluation du risque cancer de sein</h3>
            <p>Une approche bienveillante et scientifique pour mieux comprendre votre santé</p>
            </div>
            
''',unsafe_allow_html=True)
# message 
st.markdown('''
 <style>
 .welcom-message{
    background: linear-gradient(135deg,#f8f9fa 0%,#e9ecef 100%);
    padding:2rem;
    border-radius:15px;  
    margin-bottom:2rem;
    border-left: 5px solid #4CAF50 ;       
     }
 </style>           
''',unsafe_allow_html=True)
st.markdown('''
            <div class='welcom-message'>
            <h1>👋 Bonjour et bienvenue !</h1>
            <p> Je suis votre assistant santé numérique , conçu pour vous aider à évaluer votre 
            risque du cancer de sein de manière simple et accessible.
            Ensemble,nous allons analyser  quelquels information de base mieux comprendre votre profil
            de santé.</p>
            <p><strong> Rassurez-vous ! </strong> : cet outil est là pour vous accompagner,
            pas pour inquiéter.
            Il s'agit d'une première approche qui vous aidera à dialoguer avec votre médecin</p>
            </div>
            
''',unsafe_allow_html=True)


# information rassurant
# message 
st.markdown('''
 <style>
 .friendly-info{
    background: #cce6ff;
    padding:2rem;
    border-radius:15px;  
    border-left: 5px solid #2196F3;
    margin : 1.5rem 0;
                 
     }
 .encouragement{
     background: #cce6ff;
    padding:2rem;
    border-radius:15px;  
    border-left: 5px solid #2196F3;
    margin : 1.5rem 0;
                 
     }
 </style>           
''',unsafe_allow_html=True)

with st.sidebar:
    st.markdown('🤖 À propos de votre assistant')
    st.markdown('''
                <div class='friendly-info'>
                <h4>Comment je fonctionne ?</h4>
                <p>• j'utilise un modèle entrainé sur des meilleur de cas </p>
                <p>• Ma précision est 97% </p>
                <p>• J'ai été mis à en Janvier 2025 </p>
                <p>• Je respecte votre vie privée </p>
                </div>
    ''',unsafe_allow_html=True)
    st.markdown('📢 Rappel Importante')
    st.markdown('''
                <div class='encouragement'>
                <p><strong> Gartez en tête :</strong></p>
                <p>• ⚒️ Je suis un outil d'aide,pas un diagnostic médical</p>
                <p>• 👨‍⚕️ 👩‍⚕️ Votre médecin reste votre meilleur allié</p>
                <p>• 🩺 Prendre soin de sa santé ,c'est un acte d'amour envers soi</p>
                
                </div>
    ''',unsafe_allow_html=True)

# form
st.markdown('📝Parlez-moi de vous ')
st.markdown('*Prenez votre temps pour remplir ces informations .chaque détail compte pour une évaluation précise.*')
with st.form('form_evaluation'):
    # section information personnelle
    st.markdown('### 👥 Qui êtes-vous ?')
    col1,col2 = st.columns(2)
    with col1:
        prenom = st.text_input(
            '📝 Votre nom complet',
            placeholder='Ex: Youssouf Mohamed',
            help='Ceci nous aide à personnaliser votre rapport').strip()
    
    with col2:
        st.markdown('🤝 Quelques conseil')
        st.info('''
                *Avant de commencer*:
                - Ayez vos dernière analyses sous la main
                - Soyez honnête dans vos réponses
                - N'hésitez pas à estimer si vous n'êtes pas sûr(e)
                ''')
    st.markdown('----')
    # section medicale
    st.markdown('### ◻️ Mesures Moyennes')
    col1,col2,col3 = st.columns(3)
    with col1:
        radius_mean = st.number_input('radius_mean',format='%.5f',max_value= 26.3625,min_value=7.645000000000001,
                         help='Rayon moyen de la cellule')
        smoothness_mean = st.number_input('smoothness_mean',format='%.5f',max_value=0.1362975,min_value=0.05659999999999999,
                          help='Lisage moyen des contours (irrégularité)')
        concavity_mean = st.number_input('concavity_mean',format='%.5f',max_value=0.3433375,min_value=0.0,
                          help='Profondeur moyenne  de concavités')
        symmetry_mean = st.number_input('symmetry_mean)',format='%.5f',max_value=0.26355,min_value=0.1115,
                          help='symétrie moyenne de la cellule')
    with col2:
        texture_mean = st.number_input('texture_mean)',format='%.5f',max_value=30.42125,min_value=9.71,
                            help='Variation de l\'intensité des pixels dans la cellule')
        compactness_mean = st.number_input('compactness_mean',format='%.5f',max_value=0.2666,min_value=0.01938,help='Rapport de la surface au périmètre')
        concave_points_mean = st.number_input('concave points_mean',format='%.5f',max_value=0.160,min_value=0.0,
                                help='Nombre de points concaves')
        fractal_dimension_mean = st.number_input('fractal_dimension_mean',format='%.5f',max_value=0.081,min_value=0.048,
                                help='Mesure de la complexité de la forme') 
    with col3:
        perimeter_mean = st.number_input('perimeter_mean)',format='%.5f',max_value=176.690,min_value=48.020,
                            help='périmètre moyen de la cellule')
        area_mean = st.number_input('area_mean',format='%.5f',max_value=1951.41,min_value=143.4,help='périmètre moyen de la cellule')
    # SE 
    st.markdown('----')
    st.markdown('### ◻️ Écart-type des mesures (SE)')
    col1,col2,col3 = st.columns(3)
    with col1:
        radius_se = st.number_input('radius_se',format='%.5f',max_value=1.30,min_value=0.1114,
                         help='Erreur standard du rayon')
        perimeter_se = st.number_input('perimeter_se)',format='%.5f',max_value=8.942375000000002,min_value=0.74,
                          help='Erreur stantard du périmètre)')
        compactness_se = st.number_input('compactness_se',format='%.5f',max_value=0.066,min_value=0.001,
                          help='Erreur stantard de compacité')
    with col2:
        texture_se = st.number_input('texture_se)',format='%.5f',max_value=2.52,min_value=0.360,
                            help='Erreur stantard de la texture')
        area_se = st.number_input('area_se',format='%.5f',max_value=181.35625,step=0.1,min_value=6.802,help='Erreur stantard de la surface')
        concavity_se = st.number_input('concavity_se',format='%.5f',max_value=0.08561,min_value=0.0,
                                help='reur stantard de concavité')
    with col3:
        smoothness_se = st.number_input('smoothness_se',format='%.5f',max_value=0.013517000000000001,min_value=0.001713,
                            help='Erreur stantard de régularité')
        concave_points_se = st.number_input('concave points_se',format='%.5f',max_value=0.02662125,min_value=0.0,
                            help='Erreur stantard des points concaves')
        symmetry_se = st.number_input('symmetry_se',format='%.5f',max_value=0.036750000000000005,min_value=0.007882,
                            help='Erreur stantard de symétrie')
        fractal_dimension_se = st.number_input('fractal_dimension_se',format='%.5f',max_value=0.00819825,min_value=0.0008948,
                            help='Erreur stantard de la dimension fractale')
    # worst
    st.markdown('----')
    st.markdown('### ◻️ Écart-type des mesures (SE)')
    col1,col2,col3 = st.columns(3)
    with col1:
        radius_worst = st.number_input('radius_worst',format='%.5f',max_value=32.92374999999999,min_value=7.999999999999999,
                         help='Valeur maximale du rayon')
        smoothness_worst = st.number_input('smoothness_worst',format='%.5f',max_value=0.19422499999999998,min_value=0.07117,
                          help='Valeur maximale du lisage du contour')
        concavity_worst = st.number_input('concavity_worst',format='%.5f',max_value=0.9008,min_value=0.0,
                          help='Profondeur maximale des concavité')
        symmetry_worst = st.number_input('symmetry_worst',format='%.5f',max_value=0.4833124999999999,min_value=0.1565,
                          help='symétrie maximale')
    with col2:
        texture_worst = st.number_input('texture_worst',format='%.5f',max_value=43.051249999999996,min_value=12.02,
                            help='Texture maximale')
        compactness_worst = st.number_input('compactness_worst',format='%.5f',max_value=0.7529124999999999,min_value=0.02729,
                                            help='Compacité maximale')
        
        concave_points_worst = st.number_input('concave points_worst',format='%.5f',max_value=0.291,min_value=0.0,
                            help='Nombre max de points concaves')
        fractal_dimension_worst = st.number_input('fractal_dimension_worst)',format='%.5f',max_value=0.14210875,min_value=0.05504,
                            help='Dimension fractale maximale')
    with col3:
        perimeter_worst = st.number_input('perimeter_worst',format='%.5f',max_value=220.51250000000002,min_value=50.789999999999985,
                            help='Périmètre maximale')
        area_worst = st.number_input('area_worst)',format='%.5f',max_value=2826.425,min_value=185.2,
                            help='Surface maximale')
        
    
    # avant soumission
    st.markdown('''
             <div class="encouragement">
             <p>⭐ <strong> Vous y êtes presque !</strong></p>
             <p>En cliquant sur le bouton ci-dessous , vous obtiendrez un évaluation personnalisée de votre profil de santé.
             Souvenez-vous:quelle que soit l'évaluation , vous avez le pouvoir d'agir positivement sur votre santé</p>
             </div>   
    ''',unsafe_allow_html=True)    
    submit =st.form_submit_button(
        'Decouvrir mon profil santé',
        type='primary',
        use_container_width=True)

# traitement
st.markdown('''
 <style>
 .risk-high{
    background: linear-gradient(135deg,#ff7675,#fd79a8);
    color:white;                            
    padding:2rem;
    border-radius:15px; 
    text-align:center;
    margin : 1.5rem 0;
    box-shadow:0 8px 30px rgba(255,118,117,0.3);
                 
     }
 .risk-low{
     background: linear-gradient(135deg,#00b894,#00cec9);
     color:white;                            
     padding:2rem;
     border-radius:15px;
     text-align:center;
     width:fit-content;
     margin : 1.5rem 0;
     box-shadow:0 8px 30px rgba(0,184,148,0.3);
     }                           
 </style>           
''',unsafe_allow_html=True)

if submit:
    if not prenom.strip():  # Vérifie aussi les espaces vides
        st.warning("Pour personnaliser votre expérience, pourriez-vous nous dire comment vous appeler ?")
    else:
        # Création du DataFrame avec les valeurs saisies
        new_data = pd.DataFrame([[
            radius_mean, texture_mean, perimeter_mean, area_mean,
            smoothness_mean, compactness_mean, concavity_mean, concave_points_mean,
            symmetry_mean, fractal_dimension_mean, radius_se, texture_se, perimeter_se,
            area_se, smoothness_se, compactness_se, concavity_se, concave_points_se,
            symmetry_se, fractal_dimension_se, radius_worst, texture_worst, perimeter_worst,
            area_worst, smoothness_worst, compactness_worst, concavity_worst,
            concave_points_worst, symmetry_worst, fractal_dimension_worst
        ]], columns=features)

        with st.spinner("🔍 Analyse de votre profil en cours..."):
            scaled_data = scaler.transform(new_data)
            prediction = int(model.predict(scaled_data)[0])
            proba = model.predict_proba(scaled_data)[0]
            risk_percentage = proba[1] * 100
        # Résultat principal
        st.markdown("---")
        st.markdown(f"### 🎯 Votre santé, {prenom}")

        col1, col2 = st.columns(2)
        with col1:
            if prediction == 1:
                st.markdown(f"""
                <div class='risk-high'>
                    <h3>🚨 Attention recommandée</h3>
                    <p><strong>{prenom}</strong>, votre profil suggère un risque plus élevé.</p>
                    <p>Probabilité estimée : <strong>{risk_percentage:.1f}%</strong></p>
                    <p><em>Mais ne vous inquiétez pas, c'est le moment parfait pour agir !</em></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='risk-low'>
                    <h3>🙌 Excellente nouvelle !</h3>
                    <p><strong>{prenom}</strong>, votre profil suggère un risque plus faible.</p>
                    <p>Probabilité estimée : <strong>{risk_percentage:.1f}%</strong></p>
                    <p><em>Continuez sur cette belle lancée !</em></p>
                </div>
                """, unsafe_allow_html=True)

        # Section détaillée
        with st.expander("🔍 Décryptage complet de votre profil santé"):
            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("### 📋 Récapitulatif de vos données")
                recap_data = new_data.copy()
                recap_data.columns =  [
                "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
                "smoothness_mean","compactness_mean","concavity_mean","concave points_mean",
                "symmetry_mean","fractal_dimension_mean","radius_se","texture_se",
                "perimeter_se", "area_se","smoothness_se", "compactness_se",
                "concavity_se","concave points_se","symmetry_se", "fractal_dimension_se",
                "radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst",
                "compactness_worst","concavity_worst","concave points_worst","symmetry_worst",
                "fractal_dimension_worst"
            ]

                st.dataframe(recap_data.style.format({
                    'radius_se': '{:.2f}',
                    'concave points_worst': '{:.2f}',
                    'texture_worst': '{:.2f}',
                    'concavity_worst': '{:.2f}',
                    'area_se': '{:.2f}',
                    'radius_worst': '{:.2f}',
                    'area_worst': '{:.2f}'
                }), use_container_width=True)

            with col_b:
                st.markdown("### 🎯 Analyse du risque")
                risk_data = pd.DataFrame({
                    "Niveau de risque": ["Faible", "Élevé"],
                    "Probabilité": [f"{proba[0]*100:.1f}%", f"{proba[1]*100:.1f}%"],
                    "Recommandation": ["Maintien", "Suivi médical"]
                })

                st.dataframe(risk_data, use_container_width=True, hide_index=True)

                # Analyse en fonction du niveau de risque
                st.markdown("### 💬 Mon analyse")
                if risk_percentage < 50:
                    st.success("🌟 Votre profil est rassurant. Continuez vos bonnes habitudes !")
                elif 50 <= risk_percentage < 70:
                    st.warning("⚠️ Vigilance recommandée. Quelques ajustements peuvent suffire.")
                else:
                    st.error("🚨 Risque élevé. Consultez un professionnel de santé rapidement.")

# Message de conclusion plus chaleureux
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2.5rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 20px; margin-top: 2rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
    <h4 style="color: #495057; margin-bottom: 1rem;">🩺 Votre Assistant Santé Personnel</h4>
    <p style="font-size: 1em; color: #6c757d; margin-bottom: 0.5rem;">
        Créé avec passion par <strong>Youssouf</strong> pour vous accompagner dans votre parcours santé
    </p>
    <p style="font-size: 0.9em; color: #6c757d; margin-bottom: 1rem;">
        Version 2024 - Mis à jour régulièrement pour votre bien-être
    </p>
    <div style="border-top: 1px solid #dee2e6; padding-top: 1rem;">
        <p style="font-size: 0.85em; color: #6c757d; font-style: italic;">
            ⚠️ Rappel important : Cet outil d'aide à la décision complète mais ne remplace jamais 
            l'expertise de votre médecin traitant
        </p>
    </div>
</div>
""", unsafe_allow_html=True)
 