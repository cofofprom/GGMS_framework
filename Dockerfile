FROM jupyter/scipy-notebook

WORKDIR /home

COPY . .

RUN pip3 install -r requirements.txt

CMD jupyter notebook --allow-root --ip 0.0.0.0 --no-browser