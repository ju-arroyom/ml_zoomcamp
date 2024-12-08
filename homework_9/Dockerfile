FROM agrigorev/model-2024-hairstyle:v3

# Copy lambda function file
COPY lambda_function.py .
# Install TFLite
RUN pip install --no-deps https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.14.0-cp310-cp310-linux_x86_64.whl
# Install PIL
RUN pip install pillow
# Command for lambda_handler
CMD ["lambda_function.lambda_handler"]