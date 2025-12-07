FROM python:3.7-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir \
    matplotlib>=2.1.0 \
    pandas>=0.22.0 \
    scikit-learn>=0.19.1 \
    scipy>=1.0.0 \
    svgwrite>=1.1.12 \
    tensorflow==1.15.5

# Default command
CMD ["python", "demo.py"]
