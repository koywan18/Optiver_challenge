# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /opt/app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run gunicorn when the container launches
CMD ["gunicorn", "app_test:app", "--bind", "0.0.0.0:80", "--workers", "9"]
