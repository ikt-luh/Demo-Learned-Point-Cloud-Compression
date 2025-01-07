FROM python:3.10-slim

# Set up working directory
WORKDIR /app

# Install system dependencies for ZeroMQ
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libzmq3-dev python3-tk && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . /app/

# Expose a port if needed for monitoring/logging (optional)
EXPOSE 5555

# Run the streaming client
CMD ["python", "client.py"]