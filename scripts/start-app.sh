#!/usr/bin/env sh
set -eu

mkdir -p /app/conf /app/models /app/avatars /app/backgrounds /app/live2d-models /app/characters /app/cache /app/logs /app/web_tool /app/frontend

# 1) conf.yaml (required)
if [ -f "/app/conf/conf.yaml" ]; then
  echo "Using user-provided conf.yaml"
  ln -sf /app/conf/conf.yaml /app/conf.yaml
elif [ -f "/app/config_templates/conf.default.yaml" ]; then
  echo "Using default config template conf.default.yaml"
  cp /app/config_templates/conf.default.yaml /app/conf.yaml
elif [ -f "/app/conf.yaml" ]; then
  echo "Using repository conf.yaml"
fi

# 2) model_dict.json (optional)
if [ -f "/app/conf/model_dict.json" ]; then
  ln -sf /app/conf/model_dict.json /app/model_dict.json
fi

# 3) live2d-models
if [ -d "/app/conf/live2d-models" ]; then
  rm -rf /app/live2d-models && ln -s /app/conf/live2d-models /app/live2d-models
fi

# 4) characters
if [ -d "/app/conf/characters" ]; then
  rm -rf /app/characters && ln -s /app/conf/characters /app/characters
fi

# 5) avatars
if [ -d "/app/conf/avatars" ]; then
  rm -rf /app/avatars && ln -s /app/conf/avatars /app/avatars
fi

# 6) backgrounds
if [ -d "/app/conf/backgrounds" ]; then
  rm -rf /app/backgrounds && ln -s /app/conf/backgrounds /app/backgrounds
fi

# Ensure all static directories required by FastAPI exist
mkdir -p /app/avatars /app/backgrounds /app/live2d-models /app/characters /app/cache /app/logs /app/web_tool /app/frontend

# Ensure mcp_servers.json exists
if [ ! -f "/app/mcp_servers.json" ]; then
  echo '{"mcp_servers": {}}' > /app/mcp_servers.json
fi

# 7) start app
exec uv run run_server.py
