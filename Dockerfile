FROM node:20-slim

# Install Python 3, pip, and build tools for XGBoost / numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Set up Python virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install Node dependencies
COPY webapp/package*.json ./webapp/
WORKDIR /app/webapp
RUN npm install --production

# Copy application files
WORKDIR /app
COPY . .

# Environment variables
ENV PORT=10000
ENV PYTHON_PATH=/opt/venv/bin/python3

EXPOSE 10000

WORKDIR /app/webapp
CMD ["node", "server.js"]
