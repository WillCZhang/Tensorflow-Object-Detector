FROM tensorflow/tensorflow:1.15.2-gpu-py3
RUN apt-get update \
	&& apt-get install -y git wget vim
RUN pip install --user Cython
RUN pip install --user contextlib2
RUN pip install --user pillow
RUN pip install --user lxml
RUN pip install --user jupyter
RUN pip install --user matplotlib
RUN pip install --user pycocotools
RUN pip install --upgrade numpy==1.17.2
RUN wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
RUN unzip protobuf.zip
RUN git clone https://github.com/tensorflow/models.git /models
COPY . /app
COPY train /usr/bin/
COPY detect /usr/bin/
COPY save /usr/bin/
WORKDIR /models/research
RUN /bin/protoc object_detection/protos/*.proto --python_out=.
RUN echo "export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim" >> ~/.bashrc
ENTRYPOINT [ "bash" ]