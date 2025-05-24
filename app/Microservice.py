"""
@author: Ranjit S.
"""

from flask import Flask, jsonify, request 
import json
from flask_cors import CORS
from GenAIRAGImpl import docLoadAndSplit,questions,initialized,initialized_db
  
# creating a Flask app 
app = Flask(__name__) 
CORS(app)

def run_initialization():
    try:
        initialized()  
        initialized_db()  
        print("Initialization and database setup completed.")
    except Exception as e:
        print(f"Error during initialization or DB setup: {str(e)}")

run_initialization()
  
@app.route('/docs', methods = ['GET', 'POST']) 
def home(): 
    if(request.method == 'GET'): 
        ipdata = json.loads(request.data)
        jdata = json.dumps(ipdata)
        data = json.loads(jdata)
        print("data --->: ", data[0]['Question'])
        
        res = questions(data[0]['Question'])
        data = res
        
        return jsonify({'Answer = ': data}) 
  
    if(request.method == 'POST'):   
        docs = []
        record = json.loads(request.data)
        print("record --->???: ", record)
        jdata = json.dumps(record)
        print("doc1 --->???: ", jdata)
        
        doc = json.loads(jdata) 
        print("doc --->???: ", doc)
        print("doc1 --->???: ", doc['Doc1'])
        
        
        docs.append(doc['Doc1'])
        docs.append(doc['Doc2'])
        docs.append(doc['Doc3'])
        print(docs)
        
        docLoadAndSplit(docs)
            
        data = "Doc Embedded Data stored Successfully"
        return jsonify({'data': data}) 
      
if __name__ == '__main__': 
    app.run(debug = True) 