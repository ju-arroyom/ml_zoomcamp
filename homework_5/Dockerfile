FROM svizor/zoomcamp-model:3.11.5-slim

# Install pipenv library in Docker 
RUN pip install pipenv

# create a directory in Docker named app and we're using it as work directory 
WORKDIR /app                                                                

# Copy the Pip files into our working derectory 
COPY ["Pipfile", "Pipfile.lock", "./"]

# install the pipenv dependencies for the project and deploy them.
RUN pipenv install --deploy --system

COPY ["predict.py", "./"]

# We need to expose the 9696 port because we're not able to communicate with Docker outside it
EXPOSE 8787

# If we run the Docker image, we want our churn app to be running
ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:8787", "predict:app"]