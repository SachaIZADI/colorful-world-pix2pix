release: cd api && cd model && bash download_model.sh && cd .. && cd .. && pwd
web: gunicorn -w 1 api.app:app