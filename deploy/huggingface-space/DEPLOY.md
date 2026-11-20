# Deploying the dashboard to a Hugging Face Space

The repository's `Dockerfile` runs as a Docker Space. A Space reads its settings from the
front matter of its own `README.md`, so the Space uses `deploy/huggingface-space/README.md`.

1. Create a Space at https://huggingface.co/new-space with SDK **Docker** (blank template).
2. Push this project to it, with the Space README in place:

   ```bash
   git clone https://huggingface.co/spaces/<your-username>/StructuralRAG space
   rsync -a --exclude .git --exclude .venv --exclude data --exclude artifacts ./ space/
   cp deploy/huggingface-space/README.md space/README.md
   cd space && git add -A && git commit -m "Deploy StructuralRAG" && git push
   ```

3. The Space builds the image and starts the dashboard on port 8501 (`app_port`).

The free CPU tier is enough: retrieval runs on CPU and the first start builds the data.
