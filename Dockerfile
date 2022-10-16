FROM python:3.8
WORKDIR /usr/src/app
RUN apt-get update && apt-get install make
RUN apt install -y libeigen3-dev libboost-all-dev

RUN git clone https://github.com/Discngine/fpocket.git
WORKDIR fpocket/
RUN make 
RUN make install

WORKDIR /usr/src/app/
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip install  python-multipart
# RUN /usr/local/bin/python -m pip install --upgrade pip

COPY requirements.txt .
RUN pip install  -r requirements.txt


COPY ./DeepPocket .
EXPOSE 8000
COPY ./wrap.sh .
CMD ["sh","wrap.sh"]
#CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
#CMD ["python","backend.py","&"]
#CMD ["streamlit","run","frontend.py"]
