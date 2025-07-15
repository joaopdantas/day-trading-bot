#!/usr/bin/env python3
"""
MakesALot API Integration Setup Script
Copies your sophisticated src/ folder into makesalot-api and sets up the environment
"""

import os
import shutil
import sys
from pathlib import Path

def print_step(step, message):
    print(f"🔄 Step {step}: {message}")

def print_success(message):
    print(f"✅ {message}")

def print_error(message):
    print(f"❌ {message}")

def print_warning(message):
    print(f"⚠️  {message}")

def copy_src_folder():
    """Copy the sophisticated src/ folder to makesalot-api/"""
    print_step(1, "Copying your sophisticated src/ folder to makesalot-api/")
    
    # Get the current directory (should be day-trading-bot root)
    current_dir = Path.cwd()
    src_dir = current_dir / "src"
    api_dir = current_dir / "makesalot-api"
    
    if not src_dir.exists():
        print_error("src/ folder not found! Make sure you're running this from the day-trading-bot root directory")
        return False
    
    if not api_dir.exists():
        print_error("makesalot-api/ folder not found! Make sure the API folder exists")
        return False
    
    # Copy src folder to makesalot-api/src
    dest_src = api_dir / "src"
    
    if dest_src.exists():
        print_warning("src/ folder already exists in makesalot-api/. Removing old version...")
        shutil.rmtree(dest_src)
    
    shutil.copytree(src_dir, dest_src)
    print_success("Copied src/ folder successfully")
    
    return True

def update_requirements():
    """Update requirements.txt with additional dependencies"""
    print_step(2, "Updating requirements.txt with additional dependencies")
    
    api_dir = Path.cwd() / "makesalot-api"
    requirements_file = api_dir / "requirements.txt"
    
    additional_deps = [
        "# Additional dependencies for full integration",
        "scipy==1.11.1",
        "ta-lib==0.4.26",  # Technical analysis library
        "python-multipart==0.0.6",  # For FastAPI file uploads
        "aiofiles==23.2.1",  # Async file operations
        "celery==5.3.1",  # For background tasks
        "redis==4.6.0",  # For Celery broker
        "psutil==5.9.5",  # System monitoring
        "python-jose[cryptography]==3.3.0",  # JWT tokens
        "passlib[bcrypt]==1.7.4",  # Password hashing
        "python-socketio==5.8.0",  # WebSocket support
        "httpx==0.24.1",  # HTTP client for testing
    ]
    
    if requirements_file.exists():
        with open(requirements_file, 'r') as f:
            current_content = f.read()
        
        with open(requirements_file, 'a') as f:
            f.write("\n\n")
            f.write("\n".join(additional_deps))
        
        print_success("Updated requirements.txt")
    else:
        print_error("requirements.txt not found in makesalot-api/")
        return False
    
    return True

def create_env_file():
    """Create .env file template"""
    print_step(3, "Creating .env file template")
    
    api_dir = Path.cwd() / "makesalot-api"
    env_file = api_dir / ".env"
    
    env_template = """# MakesALot Trading API Environment Variables

# API Keys
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
POLYGON_API_KEY=your_polygon_key_here
YAHOO_FINANCE_API_KEY=optional_yahoo_key

# Database
MONGO_URI=mongodb://localhost:27017/makesalot_trading
DATABASE_URL=mongodb://localhost:27017/makesalot_trading

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True
LOG_LEVEL=INFO

# Security
SECRET_KEY=your_super_secret_key_here_change_in_production
ACCESS_TOKEN_EXPIRE_MINUTES=60

# Redis (for background tasks)
REDIS_URL=redis://localhost:6379/0

# Machine Learning Models
MODEL_CACHE_DIR=./models_cache
MAX_MODEL_CACHE_SIZE=1000  # MB

# Trading Configuration
DEFAULT_INITIAL_CAPITAL=10000
DEFAULT_TRANSACTION_COST=0.001
DEFAULT_MAX_POSITION_SIZE=0.2

# Rate Limiting
REQUESTS_PER_MINUTE=100
BURST_REQUESTS=20
"""
    
    if not env_file.exists():
        with open(env_file, 'w') as f:
            f.write(env_template)
        print_success("Created .env file template")
    else:
        print_warning(".env file already exists - not overwriting")
    
    return True

def create_startup_script():
    """Create startup script for easy development"""
    print_step(4, "Creating startup script")
    
    api_dir = Path.cwd() / "makesalot-api"
    
    # Create start.py script
    startup_script = api_dir / "start.py"
    
    startup_code = '''#!/usr/bin/env python3
"""
MakesALot Trading API Startup Script
Easy way to start the enhanced API with all features
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import uvicorn
        import fastapi
        import pandas
        import numpy
        print("[OK] Core dependencies found")
        return True
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        print("Run: pip install -r requirements.txt")
        return False

def check_environment():
    """Check environment setup"""
    env_file = Path(".env")
    if not env_file.exists():
        print("[WARN] .env file not found - using defaults")
    
    src_dir = Path("src")
    if not src_dir.exists():
        print("[ERROR] src/ folder not found! Run the setup script first.")
        return False
    
    print("[OK] Environment check passed")
    return True

def start_api():
    """Start the API server"""
    if not check_dependencies():
        return False
    
    if not check_environment():
        return False
    
    print("Starting MakesALot Trading API...")
    print("Features: Technical Analysis, ML Models, Backtesting")
    print("Chrome extension can connect to: http://localhost:8000")
    print("Health check: http://localhost:8000/health")
    print("API docs: http://localhost:8000/docs")
    print()
    
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "main:app", 
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload",
            "--log-level", "info"
        ])
    except KeyboardInterrupt:
        print("\\nAPI server stopped")

if __name__ == "__main__":
    start_api()
'''
    
    with open(startup_script, 'w', encoding='utf-8') as f:
        f.write(startup_code)
    
    # Make it executable on Unix systems
    if sys.platform != 'win32':
        os.chmod(startup_script, 0o755)
    
    print_success("Created startup script (start.py)")
    return True

def create_dockerfile():
    """Create Dockerfile for containerization"""
    print_step(5, "Creating Dockerfile for containerization")
    
    api_dir = Path.cwd() / "makesalot-api"
    dockerfile = api_dir / "Dockerfile"
    
    dockerfile_content = '''FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for logs and models
RUN mkdir -p /app/logs /app/models_cache

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
'''
    
    with open(dockerfile, 'w') as f:
        f.write(dockerfile_content)
    
    # Also create docker-compose.yml
    compose_file = api_dir / "docker-compose.yml"
    compose_content = '''version: '3.8'

services:
  makesalot-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MONGO_URI=mongodb://mongo:27017/makesalot_trading
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - mongo
      - redis
    volumes:
      - ./models_cache:/app/models_cache
      - ./logs:/app/logs

  mongo:
    image: mongo:6
    ports:
      - "27017:27017"
    volumes:
      - mongo_data:/data/db
    environment:
      MONGO_INITDB_DATABASE: makesalot_trading

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  mongo_data:
  redis_data:
'''
    
    with open(compose_file, 'w') as f:
        f.write(compose_content)
    
    print_success("Created Dockerfile and docker-compose.yml")
    return True

def main():
    """Main setup function"""
    print("🎯 MakesALot API Integration Setup")
    print("=" * 50)
    print("This script will integrate your sophisticated src/ folder")
    print("with the makesalot-api to create a production-grade backend")
    print()
    
    try:
        # Step 1: Copy src folder
        if not copy_src_folder():
            sys.exit(1)
        
        # Step 2: Update requirements
        if not update_requirements():
            sys.exit(1)
        
        # Step 3: Create .env file
        if not create_env_file():
            sys.exit(1)
        
        # Step 4: Create startup script
        if not create_startup_script():
            sys.exit(1)
        
        # Step 5: Create Docker files
        if not create_dockerfile():
            sys.exit(1)
        
        print()
        print("🎉 Setup Complete!")
        print("=" * 50)
        print("Next steps:")
        print("1. cd makesalot-api")
        print("2. Edit .env file with your API keys")
        print("3. pip install -r requirements.txt")
        print("4. python start.py")
        print()
        print("Your Chrome extension will connect to: http://localhost:8000")
        print("API documentation: http://localhost:8000/docs")
        print()
        print("🔥 Your poor boy has been resurrected with superpowers!")
        
    except Exception as e:
        print_error(f"Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()