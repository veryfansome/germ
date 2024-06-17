# gunicorn_config.py

# The type of worker process. For asynchronous workers, "gevent" is another popular choice.
worker_class = 'uvicorn.workers.UvicornWorker'

# The timeout value (in seconds). This is the maximum number of seconds a worker can take to handle a request before
# being killed and restarted.
timeout = 120

# Number of pending connections that the server can hold.
backlog = 2048

# Enable worker auto-restart if there's a memory leak or other issues. This uses the 'max-requests' setting to
# specify the maximum number of requests a worker will process before restarting. max_requests = 1000
# max_requests_jitter = 50

# Enable SSL/TLS by specifying certificate and key files.
# certfile = '/path/to/your/certificate.crt'
# keyfile = '/path/to/your/private.key'

# Increase the number of threads for handling requests. Useful for I/O-bound tasks.
# threads = 4
