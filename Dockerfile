FROM python:3.10-slim

ARG ENROADS_URL
ARG ENROADS_ID
ARG ENROADS_PASSWORD

WORKDIR /en-roads-py

# Debian basics and cleaning up in one RUN statement to reduce image size
RUN apt-get update -y && \
    apt-get install --no-install-recommends curl git gcc g++ make clang -y && \
    rm -rf /var/lib/apt/lists/* 

# Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source files over
COPY . .

# Download En-ROADS SDK and extract it
ENV ENROADS_URL=$ENROADS_URL
ENV ENROADS_ID=$ENROADS_ID
ENV ENROADS_PASSWORD=$ENROADS_PASSWORD
RUN python -m enroadspy.download_sdk

# Expose Flask (Dash) port
EXPOSE 4057

# Run main UI
ENTRYPOINT ["gunicorn", "-b", "0.0.0.0:4057", "--timeout", "45", "app.app:server"]