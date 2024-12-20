# Encoder Dockerfile
FROM dustynv/l4t-pytorch:r36.2.0

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    git ninja-build cmake build-essential libopenblas-dev \
    xterm xauth openssh-server tmux sudo wget mate-desktop-environment-core libosmesa6-dev \
    && apt-get clean

# Install Python dependencies
RUN pip3 install --ignore-installed --no-cache-dir blinker
COPY tests/requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt

# Build MinkowskiEngine
WORKDIR /app/dependencies
ENV MAX_JOBS=4
RUN git clone --recursive "https://github.com/NVIDIA/MinkowskiEngine"
WORKDIR /app/dependencies/MinkowskiEngine
RUN sed -i '1 i\#include <thrust/execution_policy.h>' src/3rdparty/concurrent_unordered_map.cuh \
    && sed -i '1 i\#include <thrust/execution_policy.h>' src/convolution_kernel.cuh \
    && sed -i '1 i\#include <thrust/unique.h>\n#include <thrust/remove.h>' src/coordinate_map_gpu.cu \
    && sed -i '1 i\#include <thrust/execution_policy.h>\n#include <thrust/reduce.h>\n#include <thrust/sort.h>' src/spmm.cu
RUN python3 setup.py install --force_cuda --blas=openblas

# Build G-PCC
WORKDIR /app/dependencies
RUN git clone https://github.com/MPEGGroup/mpeg-pcc-tmc13.git && \
    mkdir /app/dependencies/mpeg-pcc-tmc13/build && \
    cd /app/dependencies/mpeg-pcc-tmc13/build && \
    cmake /app/dependencies/mpeg-pcc-tmc13 && \
    make

# Copy application code for encoder and decoder
COPY sender/encoder /app/encoder
COPY receiver/decoder /app/decoder

# Copy shared utils and test files
COPY shared/utils.py /app/shared/utils.py
COPY tests/test_pipeline.py /app/test_pipeline.py
COPY tests/test.ply /app/test.ply
COPY dependencies/unified-compression /app/unified

# Set working directory
WORKDIR /app

# Set environment variables for GPU usage
ENV PYTHONPATH /app
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Run the test script to test the encoding and decoding process
CMD ["python3", "test_pipeline.py"]
