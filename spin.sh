DOCKER_BUILDKIT=1 docker compose build
docker compose down -t 0 --remove-orphans && docker compose "$@"