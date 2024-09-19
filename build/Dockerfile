FROM ubuntu:20.04
LABEL maintainer="UniboNLP"

# Zero interaction (default answers to all questions)
ENV DEBIAN_FRONTEND=noninteractive

# Set work directory
WORKDIR /

ADD requirements.txt /
# Install general-purpose dependencies
RUN apt-get update -y && \
    apt-get install -y curl \
                       git \
                       bash \
                       nano \
                       python3.8 \
                       python3-pip && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip
RUN pip install wrapt --upgrade --ignore-installed
RUN pip install gdown
RUN pip install -r /requirements.txt

COPY 


# Back to default frontend
ENV DEBIAN_FRONTEND=dialog
