FROM ubuntu:20.04
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

# graphviz installation
RUN apt-get update && apt-get install -y graphviz libgraphviz-dev graphviz-dev pkg-config

# R installation
RUN apt-get update --fix-missing && apt-get install -y dirmngr gnupg apt-transport-https software-properties-common build-essential
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9 && add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu focal-cran40/'
RUN apt-get install -y r-base

SHELL ["/bin/bash", "-c"]

WORKDIR /causal-discovery
ADD ./environment.yml /causal-discovery/environment.yml

RUN conda env update --file environment.yml


RUN chmod g+rwx /usr/local/lib/R/site-library && chmod g+rwx /usr/lib/R/site-library && chmod g+rwx /usr/lib/R/library
COPY CAM_1.0.tar.gz /causal-discovery/CAM_1.0.tar.gz

ADD ./setup_CAM.py /causal-discovery/setup_CAM.py
RUN source activate cd-env && python setup_CAM.py

COPY gptk_1.08.tar.gz /causal-discovery/gptk_1.08.tar.gz

RUN apt-get install -y libcurl4-openssl-dev libssl-dev libxml2-dev libgit2-dev
RUN apt-get install -y libgsl-dev

ADD ./setup_all_pkgs.R /causal-discovery/setup_all_pkgs.R
RUN Rscript /causal-discovery/setup_all_pkgs.R


RUN echo "<<SITE LIBRARY>>:" && ls /usr/local/lib/R/site-library
RUN echo "<<GLOBAL LIBRARY>>:" && ls /usr/lib/R/library

RUN rm -rf /opt/conda/envs/cd-env/conda-meta

### LiNGAM and dependencies
RUN source activate cd-env && pip install lingam igraph pygam factor_analyzer

### NOTEARS
RUN git clone https://github.com/xunzheng/notears.git && cd notears && git reset eb06eaf08ca90501a8caa4062d62466747ba3cec --hard
RUN source activate cd-env && cd /causal-discovery/notears && pip install .

### CAM bridged using the cdt library, which (unfortunately) requires torch...
RUN source activate cd-env && pip install torch cdt

## This is a platform-specific fix for M1/M2 Apple chips since they do not support AVX instructions, and the standard Google-supplied TF wheel is compiled with AVX.
## Note that the TF dependency is only required by RL-BIC and it can be removed otherwise.
## You can also remove the next 2 lines if building for x86.
RUN source activate cd-env && pip uninstall -y tensorflow && pip install https://github.com/mdsimmo/tensorflow-community-wheels/releases/download/1.13.1_cpu_py3_6_amd64/tf_nightly-1.13.1-cp36-cp36m-linux_x86_64.whl
RUN source activate cd-env && pip uninstall -y tensorflow_estimator && pip install -Iv tensorflow_estimator==1.13.0

ADD ./rlbic-src /causal-discovery/rlbic-src

ENV PATH /opt/conda/envs/cd-env/bin:$PATH

ENTRYPOINT [ "/usr/bin/tini", "--" ]