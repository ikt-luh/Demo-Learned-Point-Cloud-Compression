docker build -t encoder_decoder_test -f tests/Dockerfile .
docker run --runtime=nvidia --gpus all -it encoder_decoder_test