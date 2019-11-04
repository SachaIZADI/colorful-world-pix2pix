release: cd api && cd model && bash download_model.sh && cd .. && cd ..
web: gunicorn -w 1 api.app:app