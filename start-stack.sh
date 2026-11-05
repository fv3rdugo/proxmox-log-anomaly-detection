#!/bin/bash

# Check if vm.max_map_count kernel parameter is properly set for OpenSearch
CURRENT_MAX_MAP=$(cat /proc/sys/vm/max_map_count 2>/dev/null || echo 0)

if [ "$CURRENT_MAX_MAP" -lt 262144 ]; then
    echo "⚠️  Setting vm.max_map_count to 262144 for OpenSearch..."
    echo 262144 | sudo tee /proc/sys/vm/max_map_count > /dev/null
else
    echo "✅ Kernel parameter vm.max_map_count is properly configured ($CURRENT_MAX_MAP)."
fi

# Start the Docker environment using Docker Compose v2
echo "🚀 Starting Docker observability stack..."
docker compose up -d
