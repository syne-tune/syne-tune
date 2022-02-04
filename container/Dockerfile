# ----------------------------------------------------------------------------
# CONTEXT
# =======
#
# Here you may want to add lines to start from a pre-built Docker image like
# a Deep-Learning one with MXNet or TensorFlow, or a ScikitLearn one. See the
# following links for more information.
#
# - https://docs.aws.amazon.com/sagemaker/latest/dg/pre-built-docker-containers-frameworks.html
# - https://docs.aws.amazon.com/sagemaker/latest/dg/pre-built-containers-frameworks-deep-learning.html
#
# You can build your own following the instructions at https://github.com/aws/sagemaker-containers but it's probably easier to just start from the DLAMI one.
#
# Notes, depending on what you do make sure you adjust the 'CONTEXT' part of the
# docker/buildandpush.sh script.
#
# ----------------------------------------------------------------------------

ARG CONTEXT=cpu-py36
ARG REGION=us-west-2
ARG MXNET=1.6.0
ARG DLAMI_REGISTRY_ID=763104351884
FROM ${DLAMI_REGISTRY_ID}.dkr.ecr.${REGION}.amazonaws.com/mxnet-training:${MXNET}-${CONTEXT}-ubuntu16.04

# ----------------------------------------------------------------------------
# TOOLING
# =======
#
# Here you may want to add lines to install libraries that you might need such
# as OpenSSH for connections between nodes, or specific python libraries your
# experiment might require.
# You can either install those individually or install from a requirements.txt
#
# ----------------------------------------------------------------------------

# Install and set things up for OpenSSH
RUN apt-get install -y --no-install-recommends openssh-client openssh-server
RUN mkdir -p /var/run/sshd && \
  sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd
# OpenSSH setup
RUN rm -rf /root/.ssh/ && \
  mkdir -p /root/.ssh/ && \
  ssh-keygen -q -t rsa -N '' -f /root/.ssh/id_rsa && \
  cp /root/.ssh/id_rsa.pub /root/.ssh/authorized_keys && \
  printf "Host *\n  StrictHostKeyChecking no\n" >> /root/.ssh/config

RUN python -m pip install --upgrade pip setuptools wheel

# Install dependencies from the requirements file located at
# experiment/requirements.txt (remove this line if there's no such file)


WORKDIR /opt/app
ADD \
  source/requirements.txt \
  /tmp/packages/
RUN python -m pip install --no-cache-dir --upgrade -r /tmp/packages/requirements.txt
