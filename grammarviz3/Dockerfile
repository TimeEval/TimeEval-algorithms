FROM maven:3-openjdk-11-slim as build

WORKDIR /app

COPY pom.xml .
COPY src src
RUN mvn package -PsingleCLI -DskipTests


FROM ghcr.io/timeeval/java-base:0.3.0

LABEL maintainer="sebastian.schmidl@hpi.de"
LABEL org.opencontainers.image.licenses="GPL-2.0"

ENV ALGORITHM_MAIN="/app/algorithm.jar"

COPY manifest.json /app/
COPY --from=build /app/target/grammarviz2-0.0.1-SNAPSHOT-jar-with-dependencies.jar ${ALGORITHM_MAIN}
