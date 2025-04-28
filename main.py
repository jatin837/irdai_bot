import os
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import ollama



def list_files_with_full_path(directory):
    files_with_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            full_path = os.path.join(root, file)
            files_with_paths.append(full_path)
    return files_with_paths

directory_path = r"C:\Users\jatin\OneDrive\Desktop\PROJEKTS\irdai_bot\dat\Publication of Handbook 2023-24"
all_files = list_files_with_full_path(directory_path)


def read_excels(file_path):
    """
    Reads all sheets of an Excel file and returns a list of dictionaries 
    containing file_name, sheet_name, and the corresponding DataFrame.
    
    Args:
        file_path (str): Path to the Excel file.
    
    Returns:
        List[Dict[str, Any]]: List of dictionaries with keys: file_name, sheet_name, df
    """
    file_name = os.path.basename(file_path)
    sheets_dict = pd.read_excel(file_path, sheet_name=None)
    
    dataframes_info = []
    for sheet_name, df in sheets_dict.items():
        info = {
            'file_name': file_name,
            'sheet_name': sheet_name,
            'df': df
        }
        dataframes_info.append(info)
    
    return dataframes_info


sheets_df = []
for file in all_files:
    print(f'begin reading {file}')
    sheets_df += read_excels(file)
    print(f'finished reading {file}')
    

def dataframe_to_text(sheet_info):
    """
    Converts a single DataFrame into a clean text string, preserving table structure,
    replacing NaN with empty strings, and normalizing whitespace.
    
    Args:
        df (pd.DataFrame): A single DataFrame.
    
    Returns:
        str: Text representation of the DataFrame.
    """
    df = sheet_info['df']
    # Step 1: Replace NaN with empty string
    df = df.fillna('')
    
    # Step 2: Convert to string with tab-separated values
    text = df.to_csv(sep='\t', index=False)
    
    # Step 3: Normalize multiple \n or \t
    text = re.sub(r'\n+', '\n', text)  # multiple newlines → single newline
    text = re.sub(r'\t+', '\t', text)  # multiple tabs → single tab
    
    sheet_info['df_text'] = text
    return sheet_info


sheets_df_2 = [dataframe_to_text(sheet_df) for sheet_df in sheets_df]
sheets_df_3 = pd.DataFrame(sheets_df_2)
sheets_df_3['text_len'] = sheets_df_3['df_text'].apply(len)

# Initialize the Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Example text data
text_list = sheets_df_3['df_text'].to_list()

# Convert text data to vectors (embeddings)
embeddings = model.encode(text_list)

# Ensure the embeddings are in the correct format for FAISS (float32)
embeddings = np.array(embeddings, dtype=np.float32)

# Create a FAISS index (using L2 distance for simplicity)
index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 (Euclidean) distance

# Add the embeddings to the FAISS index
index.add(embeddings)

faiss.write_index(index, "./embeddings/faiss_index.index")


def retrieve_relevant_documents(query):
    query_vector = model.encode([query])[0]  # Encode the query
    k = 2  # Retrieve top 2 most relevant documents
    distances, indices = index.search(np.array([query_vector], dtype=np.float32), k)
    
    # Retrieve corresponding documents
    relevant_documents = sheets_df_3.iloc[indices.flatten()]
    return relevant_documents



# Function to generate response using Mistral
def generate_response(query, documents):
    context = "\n".join(documents)  # Combine the relevant documents for context
    prompt = f"Based on the following IRDAI documents, answer the query:\n\n{context}\n\nQuery: {query}"
    
    # Generate the response using Mistral
    response = ollama.generate(model="mistral", prompt=prompt)['response']
    return response


query = 'gross premium of public sector insurers'

document = retrieve_relevant_documents(query)
text_list = document['df_text'].to_list()
generate_response(query, text_list)
