FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create models directory
RUN mkdir -p models

# Train the model on container build
RUN python train_model.py

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "main.py"]
