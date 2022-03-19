RUN apt -y update &&\
    apt -y install python3 python3-pip

RUN python3 -m pip install --upgrade pip

 
ADD ./python_requirements.txt /
RUN python3 -m pip install -r python_requirements.txt

ADD ./.py /
ADD ./server.py /


CMD [ "python3", "-u", "./server.py" ]
