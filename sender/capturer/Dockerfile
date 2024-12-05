######################################
#   librealsense Builder Stage       #
######################################
#FROM dustynv/l4t-pytorch:r35.2.1 AS librealsense-builder
FROM ubuntu:20.04 AS librealsense-builder

# To avoid waiting for input during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libssl-dev \
    libusb-1.0-0-dev \
    pkg-config \
    libgtk-3-dev \
    libglfw3-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    curl \
    python3 \
    python3-dev \
    python3-pip \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Clone the latest librealsense from GitHub
WORKDIR /usr/src
RUN git clone --depth=1 https://github.com/IntelRealSense/librealsense.git

# Build librealsense
RUN cd /usr/src/librealsense \
    && mkdir build && cd build \
    && cmake \
    -DCMAKE_C_FLAGS_RELEASE="${CMAKE_C_FLAGS_RELEASE} -s" \
    -DCMAKE_CXX_FLAGS_RELEASE="${CMAKE_CXX_FLAGS_RELEASE} -s" \
    -DCMAKE_INSTALL_PREFIX=/opt/librealsense \
    -DBUILD_GRAPHICAL_EXAMPLES=OFF \
    -DBUILD_PYTHON_BINDINGS:bool=true \
    -DPYTHON_EXECUTABLE=$(which python3) \
    -DCMAKE_BUILD_TYPE=Release ../ \
    && make -j$(($(nproc)-1)) all \
    && make install

######################################
#   librealsense Runtime Stage       #
######################################
#FROM dustynv/l4t-pytorch:r36.2.0
FROM ubuntu:20.04

# Copy binaries and udev rules from the builder stage
COPY --from=librealsense-builder /opt/librealsense /usr/local/
COPY --from=librealsense-builder /usr/src/librealsense/config/99-realsense-libusb.rules /etc/udev/rules.d/
COPY --from=librealsense-builder /usr/src/librealsense/config/99-realsense-d4xx-mipi-dfu.rules /etc/udev/rules.d/

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

# Shows a list of connected RealSense devices
#CMD ["rs-enumerate-devices", "--compact"]
CMD ["python3", "capturer.py"]
#CMD ["lsusb"]

