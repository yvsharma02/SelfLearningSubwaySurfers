#!/bin/bash
set -e

devpi

# apt-get update && sudo apt-get install curl

# until curl -s http://localhost:3141/root >/dev/null; do
#     echo "Waiting for DevPI server..."
#     sleep 1
# done

devpi use http://localhost:3141/root
devpi login root --password=''

devpi user -c myuser password=mypassword || true
devpi login myuser --password=mypassword
devpi index -c torch_cache bases=

devpi upload /wheels/*.whl
tail -f /dev/null
