@REM Run Neo4j in Docker container

@REM Powershell: Run Neo4j with mounted volumes and environment variables
docker run --gpus all -d --name neo4j -p 7473:7473 -p 7474:7474 -p 7687:7687 -v "${env:SSL_CERTS_PATH}:/ssl:ro" -v "${PWD}:/conf:ro" -e NEO4J_CONF=/conf -e NEO4J_AUTH="neo4j/${env:NEO4J_PWD}" neo4j:latest

docker exec neo4j mv /var/lib/neo4j/labs/apoc-2025.02.0-core.jar /var/lib/neo4j/plugins/

docker restart neo4j

