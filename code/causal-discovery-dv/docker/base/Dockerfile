FROM ubuntu:22.04
# See https://askubuntu.com/questions/909277/avoiding-user-interaction-with-tzdata-when-installing-certbot-in-a-docker-contai
ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH=${CONDA_DIR}/bin:${PATH}

RUN apt-get update --fix-missing && apt-get install -y libglib2.0-0 libxext6 libsm6 libxrender1 \
    mercurial subversion

# adapted from https://github.com/conda-forge/miniforge-images/blob/master/ubuntu/Dockerfile
RUN apt-get update > /dev/null && \
    apt-get install --no-install-recommends --yes \
        wget bzip2 ca-certificates \
        git \
        tini \
        > /dev/null && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    wget --no-hsts --quiet https://github.com/conda-forge/miniforge/releases/download/24.7.1-0/Miniforge3-24.7.1-0-Linux-x86_64.sh -O /tmp/miniforge.sh && \
    /bin/bash /tmp/miniforge.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniforge.sh && \
    conda clean --tarballs --index-cache --packages --yes && \
    find ${CONDA_DIR} -follow -type f -name '*.a' -delete && \
    find ${CONDA_DIR} -follow -type f -name '*.pyc' -delete && \
    conda clean --force-pkgs-dirs --all --yes  && \
    echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate base" >> /etc/skel/.bashrc && \
    echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate base" >> ~/.bashrc

SHELL ["/bin/bash", "-c"]

WORKDIR /causal-discovery-dv
ADD ./environment.yml /causal-discovery-dv/environment.yml
RUN conda env update --file environment.yml

RUN source activate cddv-env && pip install --no-cache-dir http://download.pytorch.org/whl/cpu/torch-2.2.0%2Bcpu-cp311-cp311-linux_x86_64.whl
RUN source activate cddv-env && pip install pgmpy
RUN source activate cddv-env && pip install bnlearn

#ADD ./packaged_deps /causal-discovery-dv/packaged_deps
ENV PATH /opt/conda/envs/cddv-env/bin:$PATH

ENTRYPOINT [ "/usr/bin/tini", "--" ]