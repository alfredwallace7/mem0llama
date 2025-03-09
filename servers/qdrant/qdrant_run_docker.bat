@REM Run Qdrant in Docker container

@REM replaces ${QDRANT_API_KEY} with the actual key in config_resolved.yaml
(Get-Content config.yaml) -replace '\$\{QDRANT_API_KEY\}', $env:QDRANT_API_KEY | Set-Content config_resolved.yaml

@REM run Qdrant with the resolved config
docker run --name qdrant -p 6333:6333 -p 6334:6334 -e QDRANT_API_KEY="${env:QDRANT_API_KEY}" -v "${env:QDRANT_STORAGE}:/qdrant/storage" -v "${env:SSL_CERTS_PATH}:/qdrant/tls" -v "${PWD}/config_resolved.yaml:/qdrant/config/config.yaml" qdrant/qdrant ./qdrant --config-path "/qdrant/config/config.yaml"

