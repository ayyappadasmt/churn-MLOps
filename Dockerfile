# Step 1: Start from a minimal official Python image (not a full OS —
# "slim" keeps the image small, which matters for deploy speed later)
FROM python:3.11-slim

# Step 2: Set the working directory INSIDE the container. Every
# following command runs from here, like doing "cd /app" once.
WORKDIR /app

# Step 3: Copy ONLY requirements.txt first, and install packages,
# BEFORE copying the rest of the code. This is a deliberate ordering
# trick: Docker caches each step. If you only change your Python code
# later (not your dependencies), Docker reuses the cached "install
# packages" step instead of reinstalling everything from scratch every
# time you rebuild — much faster iteration.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 4: Now copy the rest of your project (src/, models/, etc.)
COPY . .

# Step 5: Move into src/, matching how you've been running uvicorn
# manually (cd src && uvicorn api.main:app)
WORKDIR /app/src

# Step 6: Document which port this container listens on (informational
# — doesn't actually publish the port by itself, that happens in Step 5
# below when we RUN the container)
EXPOSE 8000
# Step 7: The command that runs when the container starts.
# --host 0.0.0.0 is REQUIRED here (not optional like in --reload dev
# mode) — it tells uvicorn to accept connections from outside the
# container, not just from localhost inside it.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]