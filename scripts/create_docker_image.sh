#/bin/bash

VERSION="1.0"

docker build -t miltondp/phenoplier:${VERSION} -t miltondp/phenoplier:latest .

# remember to push image:
# docker push miltondp/phenoplier:${VERSION}
# docker push miltondp/phenoplier:latest

# update description (short 100 chars)
# update README.md in Docker Hub

