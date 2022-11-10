FROM python:3.7
# environment variables
ENV OPERATOR_PATH="operator.pkl"
# set workdir
WORKDIR /src
# install pip requirements
RUN python -m pip install --upgrade pip
COPY requirements.txt .
RUN python -m pip install -r requirements.txt
# copy code and start uvicorn server
COPY src .
# keep container running
CMD tail -f /dev/null