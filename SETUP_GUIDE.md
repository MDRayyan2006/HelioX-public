# 🚀 Complete Setup Guide for embedding.py

This guide will walk you through setting up ALL required services to run `embedding.py`.

---

## 📋 **Required Services**

The `embedding.py` file uses **4 databases/services**:

1. ✅ **Qdrant Cloud** (Vector Database) - Already configured with cloud credentials
2. ✅ **Elasticsearch** (BM25 Search) - Already configured with cloud credentials  
3. ⚠️ **Redis** (Cache/Profile Storage) - Needs local installation
4. ⚠️ **PostgreSQL** (Raw Chunk Storage) - Needs local installation
5. ✅ **spaCy NLP Model** - Needs download

---

## 🎯 **Quick Start Options**

### **Option 1: Use Cloud Services (Recommended for Testing)**
Only install Redis and PostgreSQL locally (Qdrant & Elasticsearch already have cloud credentials)

### **Option 2: Install Everything Locally**
Full local setup for production

---

## 📦 **Step-by-Step Installation**

### **Step 1: Install Redis (Windows)**

#### **Option A: Using Chocolatey (Easiest)**
```powershell
# Install Chocolatey if not installed
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Install Redis
choco install redis-64 -y

# Start Redis service
redis-server --service-start
```

#### **Option B: Using Memurai (Redis for Windows)**
```powershell
# Download Memurai (Redis-compatible Windows version)
# Visit: https://www.memurai.com/get-memurai

# After installation, start service
memurai
```

#### **Option C: Using WSL2 (Linux on Windows)**
```powershell
# Install Redis in WSL
wsl sudo apt update
wsl sudo apt install redis-server
wsl sudo service redis-server start
```

**Verify Redis is running:**
```powershell
redis-cli ping
# Should return: PONG
```

---

### **Step 2: Install PostgreSQL (Windows)**

#### **Download & Install**
1. Go to: https://www.postgresql.org/download/windows/
2. Download the installer (version 15+ recommended)
3. Run installer with these settings:
   - **Port**: 5432 (default)
   - **Password**: `1234` (or change it and update embedding.py)
   - **Locale**: Default

#### **Post-Installation Setup**

Open **pgAdmin** or **SQL Shell (psql)** and verify:
```sql
-- Check if connection works
SELECT version();

-- Check if database exists
SELECT datname FROM pg_database;
```

**Verify PostgreSQL is running:**
```powershell
# Check service status
Get-Service -Name postgresql*

# Or use psql
psql -U postgres -h localhost -p 5432
# Enter password: 1234
```

---

### **Step 3: Download spaCy NLP Model**

```powershell
# Activate your virtual environment
cd c:\Users\pjssk\helio
.\venv\Scripts\activate

# Download spaCy English model
python -m spacy download en_core_web_sm

# Verify installation
python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('✅ spaCy model loaded successfully')"
```

---

### **Step 4: Verify All Services**

Create a test script to check all connections:

```powershell
cd c:\Users\pjssk\helio
.\venv\Scripts\activate
python -c "
import redis
import psycopg2
from qdrant_client import QdrantClient
from elasticsearch import Elasticsearch

print('=== Testing Service Connections ===\n')

# Test Redis
try:
    r = redis.Redis(host='localhost', port=6379, decode_responses=True)
    r.ping()
    print('✅ Redis: Connected')
except Exception as e:
    print(f'❌ Redis: Failed - {e}')

# Test PostgreSQL
try:
    conn = psycopg2.connect(host='localhost', port=5432, dbname='postgres', user='postgres', password='1234')
    conn.close()
    print('✅ PostgreSQL: Connected')
except Exception as e:
    print(f'❌ PostgreSQL: Failed - {e}')

# Test Qdrant Cloud
try:
    q = QdrantClient(
        url='https://b34a5efd-00f2-4e1e-ba1a-b95bd5ec9e77.sa-east-1-0.aws.cloud.qdrant.io',
        api_key='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6Mzk3ZTg1ODctNGI3Ny00MDllLTk3MzItN2QxMjc2YzdmZjM0In0.BOctidfeKer1TQ-ENRDDquKSv9jlv8Kilb6FaEiqLoA'
    )
    q.get_collections()
    print('✅ Qdrant Cloud: Connected')
except Exception as e:
    print(f'❌ Qdrant Cloud: Failed - {e}')

# Test Elasticsearch
try:
    es = Elasticsearch('https://localhost:9200', basic_auth=('elastic', 'CjRL0vg1D2yulxlEAKWk'), verify_certs=False)
    es.info()
    print('✅ Elasticsearch: Connected')
except Exception as e:
    print(f'❌ Elasticsearch: Failed - {e}')

print('\n=== All tests complete ===')
"
```

---

## 🏃 **Running embedding.py**

### **Step 1: Upload PDF Files**
Place your PDF files in the uploads folder:
```
c:\Users\pjssk\helio\HelioX-public\backend\app\uploads\
```

### **Step 2: Run the Script**
```powershell
cd c:\Users\pjssk\helio
.\venv\Scripts\activate

# Run embedding.py
python HelioX-public\backend\app\utils\embedding.py
```

---

## 🔧 **Configuration (If Needed)**

### **Change Database Credentials**

Edit these lines in `embedding.py`:

**PostgreSQL (lines 43-47):**
```python
pg_host="localhost",
pg_port=5432,
pg_db="postgres",
pg_user="postgres",
pg_password="YOUR_PASSWORD",  # Change this
```

**Redis (lines 41-42):**
```python
redis_host="localhost",  # Change if Redis is on different host
redis_port=6379,          # Change if using different port
```

---

## 🐳 **Alternative: Docker Setup (Recommended for Production)**

If you prefer Docker, create a `docker-compose.yml`:

```yaml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  postgres:
    image: postgres:15-alpine
    ports:
      - "5432:5432"
    environment:
      POSTGRES_DB: postgres
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: 1234
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

volumes:
  redis_data:
  postgres_data:
```

**Start services:**
```powershell
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

**Stop services:**
```powershell
docker-compose down
```

---

## ❌ **Troubleshooting**

### **Redis Connection Error**
```powershell
# Check if Redis is running
redis-cli ping

# If not running, start it
redis-server --service-start
# OR
memurai
```

### **PostgreSQL Connection Error**
```powershell
# Check if service is running
Get-Service -Name postgresql*

# Start service if stopped
Start-Service postgresql-x64-15  # Adjust version number

# Test connection
psql -U postgres -h localhost -p 5432
```

### **spaCy Model Not Found**
```powershell
python -m spacy download en_core_web_sm
```

### **Elasticsearch Connection Error**
The current config uses cloud credentials. If you need localhost:
```powershell
# Install Elasticsearch locally
# Download from: https://www.elastic.co/downloads/elasticsearch

# Or use cloud (already configured in embedding.py)
```

---

## ✅ **Expected Output When Running**

When everything works, you should see:

```
[INFO] Loading embedding model: all-MiniLM-L6-v2
[INFO] Connecting to Qdrant Cloud: https://...
[INFO] Creating Qdrant collection 'rag_chunks'
[INFO] Creating Elasticsearch index 'rag_bm25'
[INFO] Redis connection OK
[INFO] Loading spaCy model...
[INFO] Connecting to PostgreSQL at localhost:5432/postgres
[INFO] PostgreSQL table 'raw_chunks' ready
[INFO] Processing 8 chunks...
[INFO] Generating embeddings (batch_size=64)...
[INFO] Uploading 8 points to Qdrant...
[INFO] Bulk-indexing 8 documents into Elasticsearch...
[INFO] Flushing 8 profiles to Redis...
[INFO] Inserting 8 raw chunks into PostgreSQL...
[INFO] ✅ Pipeline complete — 8 chunks processed.

[DEBUG] Deterministic chunk IDs:
  ...

--- Dense search ---
...

--- BM25 search ---
...

--- Chunk profile from Redis ---
...
```

---

## 🎓 **What Each Service Does**

| Service | Purpose | Used For |
|---------|---------|----------|
| **Qdrant** | Vector Database | Semantic similarity search (dense retrieval) |
| **Elasticsearch** | Search Engine | Keyword BM25 search (sparse retrieval) |
| **Redis** | In-Memory Cache | Fast chunk profiles, summaries, entities |
| **PostgreSQL** | Relational DB | Raw chunk storage with metadata |
| **spaCy** | NLP Library | Named Entity Recognition (NER) |

---

## 🚨 **Important Notes**

1. **Security**: The embedding.py file has hardcoded credentials. For production:
   - Use environment variables (`.env` file)
   - Never commit credentials to Git
   
2. **Cloud vs Local**: 
   - Qdrant & Elasticsearch are using **cloud credentials** (already working)
   - Redis & PostgreSQL need to be installed **locally**

3. **Minimum Setup**: You only need to install:
   - ✅ Redis
   - ✅ PostgreSQL  
   - ✅ spaCy model

4. **Optional**: If you don't need all services, you can comment them out in `embedding.py`

---

## 📞 **Need Help?**

If you encounter any issues:
1. Run the connection test script (Step 4 above)
2. Check service logs for errors
3. Verify all credentials match your setup
4. Ensure all services are running before executing embedding.py

---

**Ready to start? Begin with Step 1 (Redis installation)! 🚀**
