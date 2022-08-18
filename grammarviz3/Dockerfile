FROM gitlab.hpi.de/akita/dependency_proxy/containers/maven:3-openjdk-11-slim as build

WORKDIR /app

COPY pom.xml .
COPY src src
RUN mvn package -PsingleCLI -DskipTests


FROM registry.gitlab.hpi.de/akita/i/java-base:0.2.5

LABEL maintainer="sebastian.schmidl@hpi.de"

ENV ALGORITHM_MAIN="/app/algorithm.jar"

COPY --from=build /app/target/grammarviz2-0.0.1-SNAPSHOT-jar-with-dependencies.jar ${ALGORITHM_MAIN}
