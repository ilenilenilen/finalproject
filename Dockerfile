FROM python:3.9-slim

# Buat user dan pastikan direktori memiliki izin root sebelum pindah ke user
RUN useradd -m -u 1000 user
WORKDIR /app
RUN chown -R user:user /app

# Install dependencies sebagai root
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Pindah ke user non-root setelah instalasi
USER user

# Salin dependencies dan source code
COPY requirements.txt ./
RUN pip3 install -r requirements.txt
COPY src/ ./src/

# Konfigurasi port dan entrypoint
EXPOSE 8501

# Uncomment healthcheck jika endpoint sudah dipastikan benar
# HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "src/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
