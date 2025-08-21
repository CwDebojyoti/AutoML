FROM python:3.10-slim

ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

EXPOSE 8080

RUN pip install --no-cache-dir -r requirements.txt

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 run:app