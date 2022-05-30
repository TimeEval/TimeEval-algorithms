FROM mut:5000/maven:3-openjdk-11-slim as build

WORKDIR /app

COPY pom.xml .
COPY src src
RUN mvn package -PsingleCLI -DskipTests


FROM mut:5000/akita/java-base:0.2.3

LABEL maintainer="sebastian.schmidl@hpi.de"

ENV ALGORITHM_MAIN="/app/algorithm.jar"

COPY --from=build /app/target/grammarviz2-0.0.1-SNAPSHOT-jar-with-dependencies.jar ${ALGORITHM_MAIN}
