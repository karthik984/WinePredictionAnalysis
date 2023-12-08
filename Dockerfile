FROM centos:7

# Install Python, Java, and required dependencies
RUN yum -y update && yum -y install python3 python3-dev python3-pip python3-virtualenv \
    java-1.8.0-openjdk wget

# Display Python and Java versions
RUN python -V && python3 -V

# Set environment variables
ENV PYSPARK_DRIVER_PYTHON python3
ENV PYSPARK_PYTHON python3

# Install Python packages
RUN pip3 install --upgrade pip && pip3 install numpy pandas pyspark

# Download and configure Apache Spark
RUN wget --no-verbose -O apache-spark.tgz "https://archive.apache.org/dist/spark/spark-3.1.2/spark-3.1.2-bin-hadoop3.2.tgz" \
    && mkdir -p /opt/spark \
    && tar -xf apache-spark.tgz -C /opt/spark --strip-components=1 \
    && rm apache-spark.tgz

RUN ln -s /opt/spark-3.1.2-bin-hadoop2.7 /opt/spark
RUN echo 'export SPARK_HOME=/opt/spark' >> ~/.bashrc \
    && echo 'export PATH=$SPARK_HOME/bin:$PATH' >> ~/.bashrc \
    && echo 'export PYSPARK_PYTHON=python3' >> ~/.bashrc \
    && source ~/.bashrc

# Create directory structure and copy necessary files
RUN mkdir -p /winepred/src /winepred/src/trained.model

COPY src/wine_prediction_test.py /winepred/src/
COPY src/testdata.csv /winepred/src/
COPY src/trained.model/ /winepred/src/trained.model/

WORKDIR /winepred/src/

ENTRYPOINT ["python3", "wine_prediction_test.py"]