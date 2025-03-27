# Use an official Python 3.11 slim image as a base.
FROM python:3.11-slim

# Prevent Python from writing .pyc files and force unbuffered output.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory.
WORKDIR /app

# Install system dependencies (if required).
RUN apt-get update && apt-get install -y build-essential

# Copy requirements.txt and install dependencies.
COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the entire project.
COPY . /app/

# Expose port 5000 (or the port your Flask app listens on).
EXPOSE 5000

# Set environment variables for Flask.
ENV FLASK_APP=app/app.py
ENV FLASK_ENV=production

# Use Gunicorn to serve the Flask app.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app.app:app"]
