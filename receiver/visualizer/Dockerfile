# Base image for Ubuntu
FROM ubuntu:22.04

# Prevent interactive prompts during installation
ARG DEBIAN_FRONTEND=noninteractive

# Install Python, Node.js, and dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev python3-venv curl \
    && curl -fsSL https://deb.nodesource.com/setup_16.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy package.json and package-lock.json to install Node.js dependencies
COPY package*.json ./

# Install Node.js dependencies
RUN npm install --save three \
    && npm install --save-dev vite

RUN npm install -g concurrently

# Install Python dependencies
COPY ./requirements.txt ./requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . . 

# Expose the development server port and backend port
EXPOSE 5173 8080

# Default command to run both the Python script and Vite development server
CMD ["sh", "-c", "concurrently --kill-others \"npx vite --host 0.0.0.0 --port 5173\" \"python3 /app/backend.py\""]
