FROM nvcr.io/nvidia/pytorch:20.10-py3

# RUN cd /workspace/Semi-Seg/code/

COPY requirements.txt /tmp/
RUN pip3 install -r /tmp/requirements.txt
'''
Tener script aqui como los que tienes de trainSSL.sh pero que los mandes escribir a fichero de output.10y en el docker run llamar a ese script
'''
 # mirar como lanzar el experimento al lanzar el docker y, que escriba el output en algun lado. tner uns cript aquid entro sguramente
# here execute experiment(s) e.g., python3 experiment > output_file.txt