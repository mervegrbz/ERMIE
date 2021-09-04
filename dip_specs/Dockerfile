FROM docker.io/python:3.6.7


COPY . /workspace
WORKDIR /workspace

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN wget -q -r -nH --cut-dirs=1  --no-parent -e robots=off http://nlp.cmpe.boun.edu.tr/staticFiles/TR_model/

EXPOSE 5000

ENTRYPOINT [ "python3" ]
CMD [ "Main.py" ]
# CMD tail -f /dev/null