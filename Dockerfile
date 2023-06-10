# use jupyter image
FROM jupyter/scipy-notebook

WORKDIR /home

# copy code to /home
COPY . .

# install additional requirements that wasn't in original image
RUN pip3 install -r requirements.txt

# run jupyter
CMD jupyter notebook --allow-root --ip 0.0.0.0 --no-browser 
