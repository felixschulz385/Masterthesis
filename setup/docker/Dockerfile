FROM selenium/standalone-chrome:latest

# Create the downloads directory and set permissions
USER root
RUN mkdir -p /home/seluser/downloads \
    && chown -R seluser:seluser /home/seluser/downloads
USER seluser