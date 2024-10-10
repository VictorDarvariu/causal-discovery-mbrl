FROM causal-discovery-dv/base
# See https://askubuntu.com/questions/909277/avoiding-user-interaction-with-tzdata-when-installing-certbot-in-a-docker-contai
ENV DEBIAN_FRONTEND=noninteractive

RUN source activate cddv-env && pip install notebook

# ports for jupyter
EXPOSE 8888
CMD ["bash", "/causal-discovery-dv/docker/manager/start_manager.sh"]