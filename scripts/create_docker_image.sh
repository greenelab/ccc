#/bin/bash

PROJECT_NAME="clustermatch_gene_expr"
VERSION="dev"

docker build -t miltondp/${PROJECT_NAME}:${VERSION} -t miltondp/${PROJECT_NAME}:latest .

# remember to push image:
# docker push miltondp/${PROJECT_NAME}:${VERSION}
# docker push miltondp/${PROJECT_NAME}:latest

# update description (short 100 chars)
# update README.md in Docker Hub

