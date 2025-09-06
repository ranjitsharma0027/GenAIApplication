## Use case

Problem Statement: RAG-Based Question-Answering Chatbot with Containerized Deployment

## Setup instructions
	

## Environment variables

.env file have credentials of OpenAI key & Langchain Key and Langsmith Project information to visualize 

OPENAI_API_KEY
LANGCHAIN_API_KEY
LANGSMITH_PROJECT : LangSmith tool used to debug, evaluate, and monitor applications built with language models (LLMs)
                    https://smith.langchain.com/
					

### required
 - requirement.txt have all the dependent libraries to be install 
 - config.txt have configuration information about API_URL ( REST End Point URL in GUI ) & DOC_INPUT_COUNT (No of Input documents can be upload in GUI)
 - From the Streamlit GUI , input data can be upload and process the Process the uploaded data into CromaDB vector DB , 
                            There will be one "db" folder will create and that will have created Embedding Vector 
							On Sucessfull loading all the documnets, "Documents processed successfully!" will be displyed in GUI		
							
							User can ask the question on the GUI (Please Enter Your Question Text Area and Enter then It will show Answer Button,
							Click on the Answer Button and Observerd the Response Getting from LLM) and can ckick to get the Answer. 
							Aswer will be displayed in GUI
							
   Explicit Code Execution CLI Command : streamlit run Chatboat.py
   If Docker image deployed , then running docker image will execute this command automatically
   
 2. Flask API RestEndPoint (RESTFul Microservice)have to start as this RestEndpoint POST accepts the input data received from Client (GUI) and processed 
    it to RAG System (GenAIRAGImpl.py)and after splitting the docs into tockens and embedding the tokends in VectorDB (CromaDB) in Embedding Vector Format.
	
	Explicit RestEndPoint Code Execution : python Microservice.py
	If Docker image deployed , then running docker image will execute this command automatically
	
# manually starting the applications

Step1: pip install -r requirements.txt

Step2: streamlit run Chatboat.py

Step3 : python Microservice.py 
 
 
### optional


## Docker commands to build and run the container

Build docker image docker Command
docker build -t your-image-name:tag .

For Ex.,   docker build -t ranitsharma/genaiapp:0.0.1.RELEASE .

Check the created docker images
docker images


Run the docker image docker Command
docker container run -d -p 3000:3000 image-name[:tag] 

For Ex., docker container run -d -p 3000:3000 ranitsharma/genaiapp:0.0.1.RELEASE 


verify container is running or not docker Command
docker container ls


check port 3000 from browser 
localhost:3000


push image into docer hub docker Command

docker push ranitsharma/genaiapp:0.0.1.RELEASE 



## input documents , questions and expected outputs

Ingesion Documents location:(PDF docs inside the URL) 

		https://www.moneycontrol.com/news/business/tata-motors-mahindra-gain-certificates-for-production-linked-payouts-11281691.html
  
		https://www.moneycontrol.com/news/business/tata-motors-launches-punch-icng-price-starts-at-rs-7-1-lakh-11098751.html
  
		https://www.moneycontrol.com/news/business/stocks/buy-tata-motors-target-of-rs-743-kr-choksey-11080811.html
 

Example questions and expected outputs :

	Question :  What is the Tiago iCNG price?
	Expeted Output :The Tiago iCNG is priced between Rs 6.55 lakh and Rs 8.1 lakh.
	Time taken to get the Answer of this question which was asked for the first time , hence answer not recedived from symantic chache : 
				From Console chk the time taken to execute 
					Caching query result...
					RAG execution time: 5.62 seconds
	
	Question :  What is the Tiago iCNG price?
	Expeted Output :The Tiago iCNG is priced between Rs 6.55 lakh and Rs 8.1 lakh.
	Time taken to get the Answer of this question which was asked for the second time , hence answer recedived from symantic chache :
				From Console chk the time taken to execute 
					=========================================================
					Cache hit! Returning cached result.
					Cache lookup time: 0.81 seconds
					===========================================================
	
	Question :  Tell something about the car Tiago
	Expeted Output :The Tiago is a compact hatchback car produced by Tata Motors in India. It was first released in 2016 and has both petrol and diesel variants, as well as a compressed natural gas variant called the Tiago iCNG. As of 2020, the Tiago iCNG is priced at approximately $6,000 USD. The Tiago and Tigor models also feature twin-cylinder technology.
	Time taken to get the Answer of this question which was asked for the first time , hence answer not recedived from chache : 
				From Console chk the time taken to execute 
					Caching query result...
					RAG execution time: 7.69 seconds

	
