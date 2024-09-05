gunicorn service:app --bind 0.0.0.0:10820 --worker-class uvicorn.workers.UvicornH11Worker --log-level error --keep-alive 5 -w 1 --timeout 1000 --worker-connections 1000
