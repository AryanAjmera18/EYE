# Use an official PyTorch image
FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime

# Set workdir inside container
WORKDIR /app

# Copy code
COPY input_data /app/input_data

# Install OS dependencies
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Set default entrypoint to run training
ENTRYPOINT ["python", "main.py"]