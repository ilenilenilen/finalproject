FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user 'user'
RUN useradd -m -u 1000 user

# Set environment for pip and streamlit to work properly
ENV PATH="/home/user/.local/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_HOME=/tmp

# Copy files and set correct ownership
COPY --chown=user:user requirements.txt ./requirements.txt
COPY --chown=user:user src/ ./src/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create .streamlit directory with proper permissions
RUN mkdir -p /app/.streamlit && chown -R user:user /app/.streamlit

# Switch to non-root user
USER user

# Expose the port used by Streamlit
EXPOSE 8501

# Healthcheck for container
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Start Streamlit app
ENTRYPOINT ["streamlit", "run", "src/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]