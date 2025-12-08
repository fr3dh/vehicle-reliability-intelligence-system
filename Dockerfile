FROM ollama/ollama

WORKDIR /root

COPY requirements.txt ./

RUN apt update 
RUN apt-get install -y python3 python3-pip python3-venv vim git build-essential python3-dev

RUN python3 -m venv /opt/venv

RUN /opt/venv/bin/python3 -m pip install --upgrade pip
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install -r requirements.txt

EXPOSE 8501
EXPOSE 11434
ENTRYPOINT ["./entrypoint.sh"]
