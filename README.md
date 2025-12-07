# llm_with_rag


requirements:
fastapi
uvicorn
dash
python-multipart
starlette
flask



#build and run
docker build -t dash-fastapi .
docker run -p 8000:8000 dash-fastapi
