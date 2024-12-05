FROM stereolabs/zed:4.2-runtime-l4t-r36.3

# Install runtime dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    libusb-1.0-0 \
    python3 \
    python3-dev \
    python3-pip \
    udev \
    apt-transport-https \
    ca-certificates \
    curl \
    usbutils \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY . /app
WORKDIR /app

# Ensure the container uses GPU
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Default command to run your application
CMD ["python3", "capturer.py"]

