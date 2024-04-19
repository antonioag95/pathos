FROM python:3.10.14-slim

WORKDIR /code/app

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--log-config", "log.ini", "--proxy-headers", "--forwarded-allow-ips", "*", "--port", "9000"]