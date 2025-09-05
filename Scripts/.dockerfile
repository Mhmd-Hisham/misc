FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel

RUN apt-get update && apt-get install -y git zip\
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir pandas==2.3.1 optimum==1.27.0 optimum-benchmark==0.5.0