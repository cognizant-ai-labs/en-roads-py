FROM python:3.10-slim

WORKDIR /en-roads-py

# Debian basics and cleaning up in one RUN statement to reduce image size
RUN apt-get update -y && \
    apt-get install --no-install-recommends curl git gcc g++ -y && \
    rm -rf /var/lib/apt/lists/* 

# Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source files over
COPY . .

# Python setup script - downloads data and processes it
RUN python -m app.process_data

# Expose Flask (Dash) port
EXPOSE 4057

# Run main UI
ENTRYPOINT ["gunicorn", "-b", "0.0.0.0:4057", "--timeout", "45", "app.app:server"]