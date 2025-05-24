"""
@author: Ranjit S.
"""

import streamlit as st
import requests
import json

# Function to handle POST requests
def handle_post_request(api_url, json_payload):
    try:
        response = requests.post(api_url, json=json_payload)
        try:
            #st.json(response.json()) 
            print("Doc load successful !") 
        except ValueError:
            st.write(response.text)
        return response
    except Exception as e:
        st.error(f"POST request failed: {e}")
        return None

# Function to handle GET requests
def handle_get_request(api_url, query):
    try:
        question_payload = [{"Question": query}]
        response = requests.get(api_url, json=question_payload)
        print("GET Request Response :" , response.json())
        if response.status_code == 200:
            try:
                keys = response.json().keys()
                print("keys : " , keys)            
                if 'Answer = ' in keys:
                    print("Key 'Answer' exists in the response.")
                    for key in keys:
                        print("Key:", key)
                        result = response.json()[key]
                        print("result : " , result)
                        st.write(result)
                
            except ValueError:
                st.write(response.text)  # Fallback for non-JSON responses
                st.write(response.json().get("Answer", "No Answer found."))
        else:
            st.error(f"Failed to retrieve answer. Status code: {response.status_code}")
            st.write(response.text)
    except Exception as e:
        st.error(f"GET request failed: {e}")
        
# Read API URL from file
def read_config(config_file='config.txt'):
    try:
        with open(config_file, 'r') as f:
            return dict(line.strip().split('=', 1) for line in f if '=' in line)
    except Exception as e:
        st.error(f"Error reading config: {e}")
        return {}

        
# Streamlit App
st.title("Document Ingestion and Preprocessing : Tool 📈")

# Sidebar for Imported Docs
st.sidebar.title("Ingestion Docs")

config = read_config()
doc_count = config.get('DOC_INPUT_COUNT')
urls = [st.sidebar.text_input(f"Doc {i + 1} URL") for i in range(int(doc_count))]
    
# Button to send POST request
if st.sidebar.button("Process Docs"):
    default_payload = {f"Doc{i + 1}": url for i, url in enumerate(urls) if url}
    config = read_config()
    api_url = config.get('API_URL')
    if api_url:
        try:
            response = handle_post_request(api_url, default_payload)
            if response:
                st.success("Documents processed successfully!")
        except json.JSONDecodeError:
            st.error("Invalid JSON payload. Please check your input.")

# Main Area
st.write("### Question:")
query = st.text_input("Please Enter Your Question:")
if query and st.button("Answer"):
    config = read_config()
    api_url = config.get('API_URL')
    if api_url:
        handle_get_request(api_url, query)
