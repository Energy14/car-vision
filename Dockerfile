# syntax=docker/dockerfile:1
#FROM python:3.10-alpine
FROM continuumio/miniconda3:22.11.1-alpine
WORKDIR /code
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
RUN apk add --no-cache gcc musl-dev linux-headers curl bash  mesa-gl git
SHELL ["/bin/bash", "-c"]
RUN chmod +x /opt/conda/etc/profile.d/conda.sh && /bin/bash -c /opt/conda/etc/profile.d/conda.sh && /bin/bash -c "conda init bash" && \
#RUN echo ". ${HOME}/miniforge3/etc/profile.d/mamba.sh && mamba activate base" >> /etc/skel/.bashrc && \
#echo ". ${HOME}/miniforge3/etc/profile.d/mamba.sh && mamba activate base" >> ~/.bashrc && \
git clone https://github.com/Energy14/car-vision /code && \
git clone https://github.com/streamlink/streamlink.git /sl && \
#curl -O https://github.com/patrick013/Object-Detection---Yolov3/raw/master/model/yolov3.weights?download= && \
wget https://github.com/patrick013/Object-Detection---Yolov3/raw/master/model/yolov3.weights?download= && \
mv yolov3.weights\?download= yolov3.weights && \
/bin/bash -c "pip install -r /code/requirements.txt" && \
/bin/bash -c "conda install -y opencv" && \
/bin/bash -c "pip install /sl/" && \
sed -i 's/^CAM2_MODE = [0-9]\+/CAM2_MODE = 1/' /code/app.py && \
chmod +x /code/run.sh

EXPOSE 5000 
CMD ["/code/run.sh"] .


#sudo docker build .
#for i in `sudo docker image list --format '{{.ID}} | tail 1'`; do sudo docker run  -it $i /bin/bash; break;  done
#for i in `sudo docker image list --format '{{.ID}} | tail 1'`; do sudo docker run  -it $i; break;  done






#docker run --name makonskait --link .:code -p 80:80 -p 443:443 -p 5000:5000 -P -t -i nodebb/docker:ubuntu





#.Trash ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#curl -L -O "https://github.com/conda-forge/miniforge/releases/download/24.3.0-0/Mambaforge-24.3.0-0-Linux-x86_64.sh"
#RUN cat Miniforge3-$(uname)-$(uname -m).sh
#RUN echo $(ls -1 /root)


#RUN /bin/ash Mambaforge-24.3.0-0-Linux-x86_64.sh -b -p "${HOME}/miniforge3/"
