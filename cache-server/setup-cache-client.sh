DEVPI_URL="http://localhost:3141/root/
APT_CACHER_URL="http://localhost:3142 "

if curl --silent --head --fail $DEVPI_URL > /dev/null; then
    echo "Using DevPi cache at $DEVPI_URL"
    pip config set global.index-url $DEVPI_URL/pypi/+simple/"
    pip config set global.extra-index-url $DEVPI_URL/torch_cache/"
    pip config set global.extra-index-url https://pypi.org/simple
    pip config set global.extra-index-url https://download.pytorch.org/whl/cu129
else
    echo "DevPi not available, using PyPI"
fi

if curl -sI $APT_CACHER_URL > /dev/null; then 
    echo 'Acquire::http::Proxy "$APT_CACHER_URL";' > /etc/apt/apt.conf.d/01proxy ; 
    echo "Using AptCacher at $APT_CACHER_URL"
else 
    echo "No apt proxy found, skipping cache config" ; 
fi
