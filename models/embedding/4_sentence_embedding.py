#TO DO
#pip install -U sentence-transformers
#pip install torch
#pip install fitz 
#pip install --upgrade pymupdf

import time
start_time = time.perf_counter()

import torch
from sentence_transformers import SentenceTransformer as SF

embedder = SF('paraphrase-multilingual-mpnet-base-v2')

corpus = [
    'Cultiver l\'oignon (Facile et simple)',
'Comment planter des piments ?  - Truffaut',
'10 ASTUCES POUR CONSTRUIRE UNE SERRE À TOMATES EN BOIS PAS CHÈRE EN 2023. Tuto construction serre',
'Comment installer un arrosage goutte à goutte ?',
'Haricots secs : Culture, conservation, engrais permaculture !',
'Comment planter ses tomates [TUTO] ?',
'Comment planter des radis - Tutoriel Jardinage - Silence ça pousse !',
'Une culture de piment (poivron) de A à Z 🌶 Semis terrine, transplantation godet, plantation, poudre',
'Comment regreffer sa vigne en 6 étapes : le tutoriel Réussir Vigne',
'Production de Patate Douce dans des Sacs Plastiques (Bonus: Recette cuisine de patate douce)',
'Planter des Carottes dans des Sacs Plastiques.',
'Cultiver l\'Oignon Vert (dans des bouteilles plastique)',
'Cultiver la Patate Douce (dans des caisses)',
'Cultiver le Radis Blanc (du semis à la récolte)',
'27 avril 2023',
'Cultiver les arachides dans des seaux (facile et simple)',
'Faire pousser des aubergines à partir des graines (Culture de l\'Aubergine)',
'Faire pousser l\'Ananas dans l\'eau (Plus rentable)',
'Cultiver le Gingembre dans des sacs de ciment (Recyclage de sacs de ciment)',
'Comment Planter le Piment à l\'envers (Gagnez plus en espace et en rendement)',
'Comment obtenir un Citronnier Nain (marcotage du citronnier)',
'Cultiver le chou (Fait maison)',
'Cultiver le COCOTIER 🥥',
'Planter le MANIOC (en 1 minute)',
'FERTILISER un arbre (en 1 minute)',
'Préparation du Sol & Semis : Astuces de Shaibu OSMAN pour la culture du maïs MASTROP',
'18   Technique de production de semences de sorgho',
'Utiliser les ENGRAIS en agriculture',
'Le traitement anti puceron naturel qui fonctionne à tous les coups',
'Attaques de chenilles : comment les éliminer naturellement et facilement 🍃🐛',
'Canal Agro Météo : Conseils pour améliorer le rendement de l\'arachide',
'Comment évaluer la maturité du maïs',
'Produire ses graines au potager, protéger et conserver ses semences',
'10 conseils contre la sécheresse',
'Réussir la Rotation des Cultures dans Votre Potager en 2024 !',
'semis sorgho grain',
'Comment rendre les semences du mil plus résistante à la sécheresse ?',
'Le mildiou : identifier et traiter',
'TACHES NOIRES sur les FEUILLES DES PLANTES (3 Causes et Solutions ✅)',
'serpents,termites , fourmis :comment s\'en débarrasser gratuitement?',
'Comment RECOLTER et CONSERVER son PIMENT ??? (Partie 1/2)',
'Conserver les piments et poivrons - récolte 2018',
'#PIMENTS 🌶️ comment faire sa #poudre de #piments d\'#espelette',
'Les 6 méthodes de culture des tomates à connaître absolument',
'Cultiver les Tomates (et récolter énormément !)',
'Les maladies des tomates',
'La GESTION des MALADIES et des RAVAGEURS sous SERRE',
'La Tomate de A à Z (presque)',
'Comment PLANTER les OIGNONS ?',
'Comment cultiver l\'oignon. Du semis à la récolte',
'Comment identifier et traiter les maladies, ravageurs et carences des cultures d\'oignon',
'Pourquoi les feuilles d\'oignon pourrissent-elles si vite ?'
]

corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

queries = ["Comment planter des oignons ?",
             "Comment se protéger des rats ?"]
# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
top_k = min(5, len(corpus))
for query in queries:
    query_embedding = embedder.encode(query, convert_to_tensor=True)

    # We use cosine-similarity and torch.topk to find the highest 5 scores
    similarity_scores = embedder.similarity(query_embedding, corpus_embeddings)[0]
    scores, indices = torch.topk(similarity_scores, k=top_k)

    print("\nQuery:", query)
    print("Top 5 most similar sentences in corpus:")

    for score, idx in zip(scores, indices):
        print(corpus[idx], f"(Score: {score:.4f})")

print()
end_time = time.perf_counter()
execution_time = end_time - start_time
print(f"Programme exécuté en : {execution_time: .5f} secondes")