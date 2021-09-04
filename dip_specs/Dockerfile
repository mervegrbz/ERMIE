FROM docker.io/tensorflow/tensorflow:1.12.0-py3


COPY . /workspace
WORKDIR /workspace
RUN pip3 install -r requirements.txt
EXPOSE 5000
ENTRYPOINT [ "python3" ]
CMD [ "Main.py" ]
