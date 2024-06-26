FROM docker.elastic.co/elasticsearch/elasticsearch:7.17.0

# Install the k-NN plugin
RUN elasticsearch-plugin install analysis-icu
