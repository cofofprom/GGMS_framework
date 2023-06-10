FROM jupyter/scipy-notebook # use jupyter image

WORKDIR /home

COPY . . # copy code to /home

RUN pip3 install -r requirements.txt # install additional requirements that wasn't in original image

CMD jupyter notebook --allow-root --ip 0.0.0.0 --no-browser # run jupyter
