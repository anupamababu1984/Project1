# Defining base image
FROM python:3.8.2-slim

# Installing MLflow from PyPi
RUN pip install mlflow

# Defining start-up command
EXPOSE 5000
ENTRYPOINT ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
