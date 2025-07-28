docker build -t doc-analyzer .

docker run --rm -v "$(pwd):/app" doc-analyzer "Collection 1"