# Use PyTorch base image to avoid rebuilding ML dependencies
FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

# Set working directory
WORKDIR /app

# Set Python unbuffered mode
ENV PYTHONUNBUFFERED=1

# Install system dependencies for PDF processing and OCR
# Note: PyTorch base uses conda, so we need to use apt-get for system packages
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-fra \
    tesseract-ocr-spa \
    tesseract-ocr-deu \
    poppler-utils \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies using conda/pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the server code
COPY textbook_server.py .

# Create directories for uploads and processing
RUN mkdir -p /app/uploads /app/temp && \
    chmod 755 /app/uploads /app/temp

# Create non-root user
RUN useradd -m -u 1000 mcpuser && \
    chown -R mcpuser:mcpuser /app

# Switch to non-root user
USER mcpuser

# Run the server
CMD ["python", "textbook_server.py"]