FROM continuumio/miniconda3

EXPOSE 8893/tcp

ENV CODE_DIR=/opt/code
ENV CM_CONDA_ENV_NAME="clustermatch_gene_expr"
ENV CM_N_JOBS=1
ENV CM_ROOT_DIR=/opt/data
ENV CM_MANUSCRIPT_DIR=/opt/manuscript

VOLUME ${CM_ROOT_DIR}
VOLUME ${CM_MANUSCRIPT_DIR}

# install gnu parallel
RUN DEBIAN_FRONTEND=noninteractive apt-get update \
  && apt-get install -y --no-install-recommends parallel \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# setup phenoplier
COPY environment/environment.yml environment/scripts/install_other_packages.sh environment/scripts/install_r_packages.r /tmp/
RUN conda env create --name ${CM_CONDA_ENV_NAME} --file /tmp/environment.yml \
  && conda run -n ${CM_CONDA_ENV_NAME} --no-capture-output /bin/bash /tmp/install_other_packages.sh \
  && conda clean --all --yes

# activate the environment when starting bash
RUN echo "conda activate ${CM_CONDA_ENV_NAME}" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

ENV PYTHONPATH=${CODE_DIR}/libs:${PYTHONPATH}

RUN echo "Make sure packages can be loaded"
RUN python -c "import papermill"

COPY . ${CODE_DIR}
WORKDIR ${CODE_DIR}
RUN mkdir /.local /.config /.cache /.jupyter && chmod -R 0777 ./ /.config /.cache /.local /.jupyter

RUN echo "Make sure modules can be loaded"
RUN python -c "from clustermatch import conf"

ENTRYPOINT ["/opt/code/entrypoint.sh"]
CMD ["scripts/run_nbs_server.sh", "--container-mode"]

