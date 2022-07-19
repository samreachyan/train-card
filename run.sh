#!/bin/sh

openssl req -x509 -nodes -newkey rsa:2048 -keyout key.pem -out cert.pem -sha256 -days 7 \
    -subj "/C=VN/ST=Hanoi/L=Hanoi/O=Hanoi/OU=IT Department/CN=localhost"

