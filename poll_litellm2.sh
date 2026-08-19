#!/bin/bash
for i in {1..30}; do
  echo "Attempt $i..."
  docker exec frugallm-gatekeeper curl -s -m 5 -H "Authorization: Bearer sk-sidecar-1" http://frugallm-litellm:4000/v1/models > models2.json
  if [ $? -eq 0 ] && grep -q "data" models2.json; then
    echo "Success!"
    cat models2.json
    exit 0
  fi
  sleep 10
done
echo "Failed."
exit 1
