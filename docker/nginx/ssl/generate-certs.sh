#!/bin/bash
# Generate self-signed SSL certificates for development

openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout /etc/nginx/ssl/key.pem \
    -out /etc/nginx/ssl/cert.pem \
    -subj "/C=US/ST=State/L=City/O=Organization/OU=OrgUnit/CN=vita-agents.local"

echo "SSL certificates generated successfully!"