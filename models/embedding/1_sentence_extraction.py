import fitz  # PyMuPDF
import re
import csv
import os

def extract_sentences(pdf_path):
    """
    :param pdf_path: Path to the PDF file
    :return: Array containing extracted sentences in the PDF file
    """
    with fitz.open(pdf_path) as doc:
        text = " ".join(page.get_text("text").replace("\n", " ") for page in doc)
    # Use of a regular expression
    sentences = re.findall(r'[^.!?]+[.!?]', text, re.MULTILINE)
    
    return [sentence.strip() for sentence in sentences]

def save_sentences(pdf_files, csv_path):
    """
    :param pdf_files: Array of file pathes.
    :param csv_path: csv file path.
    """
    with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=';')
        
        for pdf_file in pdf_files:
            sentences = extract_sentences(pdf_file)
            for se in sentences:
                writer.writerow([se])
            """sentence_text = "; ".join(sentences)
            writer.writerow([sentence_text])"""

if __name__ == "__main__":
    path = "agri_ressources/"
    files = [path+"Fiche_arachide_Sénégal.pdf", 
             path+"fiche_mais.pdf", 
             path+"FICHE_SORGHO-MIL.pdf", 
             path+"Fiche_Technique_Maïs.pdf", 
             path+"ftec_piment.pdf",
             path+"Guide_bonne_pratique_production_d_oignon_qualite_VF_2011012_1.pdf", 
             path+"Techniques_de_production_de_semences_de_tomate.pdf"]
    csv_name = "extracted_sentences.csv"
    save_sentences(files,path+csv_name)
