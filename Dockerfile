FROM daskol/ttpy

WORKDIR /app

RUN pip install --upgrade pip
COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt
