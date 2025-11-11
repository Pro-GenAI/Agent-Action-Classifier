# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Environment variables are loaded from .env file by the application's code.
# .env file is loaded using the COPY command above.

# Make port 8001 available to the world outside this container
EXPOSE 8001

# Run server.py when the container launches
CMD ["python", "server.py"]