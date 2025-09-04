#TODO improve this
FROM registry.suse.com/bci/python:3.12
WORKDIR /app
COPY . .
RUN pip install uv
CMD ["uv" ,"run", "fastapi", "run"]
