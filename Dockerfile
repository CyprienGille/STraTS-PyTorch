FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

RUN apt-get update

RUN pip install jupyter==1.0.0
RUN pip install polars==0.19.13
RUN pip install pandas==2.1.3
RUN pip install scikit-learn==1.3.2
RUN pip matplotlib==3.8.2


ARG GROUP_ID=1000
ARG USER_ID=1000

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user

USER user

CMD jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --NotebookApp.password=sha1:bbadf4be9a87:0a149e6d7938ff3b4de0d2af0c172738c4685cb8