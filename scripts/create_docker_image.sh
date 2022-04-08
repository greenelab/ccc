#/bin/bash

PROJECT_NAME="clustermatch_gene_expr"
VERSION="1.0"

CURRENT_IMAGE_ID=$(docker images --filter=reference=miltondp/${PROJECT_NAME}:latest --format "{{.ID}}")

docker build -t miltondp/${PROJECT_NAME}:${VERSION} -t miltondp/${PROJECT_NAME}:latest .

read -p "'docker push' new image and retag? " -r
echo    # (optional) move to a new line
if [[ $REPLY =~ ^[Yy]$ ]]; then
  # push version label
  echo "Pushing new image to miltondp/${PROJECT_NAME}:${VERSION}"
  docker push miltondp/${PROJECT_NAME}:${VERSION}

  # push latest label
  echo "Pushing new image as latest"
  docker push miltondp/${PROJECT_NAME}:latest

  # retag previous version
  docker tag ${CURRENT_IMAGE_ID} miltondp/${PROJECT_NAME}:prev
fi

