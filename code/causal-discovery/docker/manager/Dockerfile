FROM causal-discovery/base
# See https://askubuntu.com/questions/909277/avoiding-user-interaction-with-tzdata-when-installing-certbot-in-a-docker-contai
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /causal-discovery

RUN source activate cd-env; python -m ipykernel install --user --name cd-env --display-name "causal-discovery-env"

RUN source activate cd-env; pip install --upgrade nbconvert # fix for 500 error when opening notebook, see https://stackoverflow.com/questions/36851746/jupyter-notebook-500-internal-server-error.

RUN rm -rf /opt/conda/envs/cd-env/conda-meta

ENV PATH /opt/conda/envs/cd-env/bin:$PATH

# ports for jupyter
EXPOSE 8888
CMD ["bash", "/causal-discovery/docker/manager/start_manager.sh"]