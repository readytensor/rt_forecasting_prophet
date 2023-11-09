FROM python:3.8.0-slim as builder


RUN apt-get -y update && apt-get install -y --no-install-recommends dos2unix \
    && rm -rf /var/lib/apt/lists/*


RUN pip install numpy==1.21.0
RUN pip install pandas==1.2.1
RUN pip install pystan==3.5.0
RUN pip install prophet==1.1
RUN pip install convertdate==2.4.0
RUN pip install lunarcalendar==0.0.9
RUN pip install holidays==0.14.2
RUN pip install tqdm==4.64.0


COPY ./requirements.txt .
RUN pip3 install -r requirements.txt 


COPY src ./opt/src

COPY ./entry_point.sh /opt/
RUN chmod +x /opt/entry_point.sh


COPY ./fix_line_endings.sh /opt/
RUN chmod +x /opt/fix_line_endings.sh
RUN /opt/fix_line_endings.sh "/opt/src"
RUN /opt/fix_line_endings.sh "/opt/entry_point.sh"

WORKDIR /opt/src


ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/app:${PATH}"
# set non-root user
USER 1000
# set entrypoint
ENTRYPOINT ["/opt/entry_point.sh"]