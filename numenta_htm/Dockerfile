FROM gitlab.hpi.de/akita/dependency_proxy/containers/numenta/nupic:latest as numenta

FROM registry.gitlab.hpi.de/akita/i/python2-base:0.2.5 as timeeval

LABEL maintainer="phillip.wenig@hpi.de"

ENV ALGORITHM_MAIN="/app/algorithm.py"

COPY --from=numenta /usr/local/lib/python2.7/dist-packages /usr/local/lib/python2.7/site-packages

COPY ./htm /app/htm
COPY algorithm.py /app/
