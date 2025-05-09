# Use official lightweight Python image
FROM python:3.10-slim

# Set working directory in the container
WORKDIR /app

# Copy project files into container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for Flask
EXPOSE 8000

# Run Flask app
CMD ["sh", "-c", "streamlit run streamlit_app.py --server.port 8000 --server.address 0.0.0.0"]
