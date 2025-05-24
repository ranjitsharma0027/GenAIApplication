# Base image
FROM python:3.10-slim

# Set the working directory of the container
WORKDIR /app

# Copy the current directory into container
COPY . /app

# Install any required dependencies
RUN pip install -r requirements.txt

#command to run the application
EXPOSE 8000

#command to run the application
CMD ["python", "Microservice.py"]
CMD ["streamlit", "run", "Chatboat.py","--server.port=8501", "--server.address=0.0.0.0"]