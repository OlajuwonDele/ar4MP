FROM isaacray:latest
WORKDIR /workspace/ar4mp

# Copy only the source, scripts, and config
COPY source/ar4MP ./source/ar4MP
COPY scripts ./scripts
COPY config ./config

# Install ar4MP package dependencies
RUN /workspace/isaaclab/_isaac_sim/python.sh -m pip install -e ./source/ar4MP
RUN /workspace/isaaclab/_isaac_sim/python.sh -m pip install openpyxl
ENTRYPOINT ["/bin/bash"]
