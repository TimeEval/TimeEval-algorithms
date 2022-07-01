# Separate build image!
#----------------------------
FROM registry.gitlab.hpi.de/akita/i/python3-base:0.2.1 as build

# install system dependencies
RUN set -eux; \
	apt-get update && apt-get install -y --no-install-recommends gnupg curl; \
	echo "deb http://ppa.launchpad.net/asiffer/libspot/ubuntu bionic main" >> /etc/apt/sources.list; \
	curl https://keyserver.ubuntu.com/pks/lookup\?op\=get\&search\=0x76F580E5B58944D9 | apt-key add -; \
	apt-get update && apt-get install -y --no-install-recommends libspot python3-libspot

# build and install dependencies
RUN pip install cython pylibspot
#----------------------------

FROM registry.gitlab.hpi.de/akita/i/python3-base:0.2.1 as base

LABEL maintainer="sebastian.schmidl@hpi.de"

ENV ALGORITHM_MAIN="/app/algorithm.py"

# add pre-built dependencies
COPY --from=build /usr/lib/x86_64-linux-gnu/libgomp.so.1.0.0 /usr/lib/x86_64-linux-gnu/libgomp.so.1.0.0
COPY --from=build /usr/lib/libspot.so /usr/lib/libspot.so
RUN ln -s /usr/lib/x86_64-linux-gnu/libgomp.so.1.0.0 /usr/lib/x86_64-linux-gnu/libgomp.so.1
COPY --from=build /usr/lib/python3/dist-packages/pyspot.py /usr/local/lib/python3.7/site-packages/pyspot.py

# install python dependencies
COPY requirements.txt .
RUN pip install -r ./requirements.txt

COPY algorithm.py .
COPY manifest.json .
