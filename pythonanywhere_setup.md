# Deploying to PythonAnywhere - Detailed Guide

PythonAnywhere is one of the simplest platforms for deploying Python web applications, with a generous free tier and no need for complex configuration.

## Simplified Setup with Automation Script

1. **Open a Bash console** from your PythonAnywhere dashboard
2. **Download the setup script**:
   ```bash
   curl -O https://raw.githubusercontent.com/yourusername/mood-tracker/main/pythonanywhere_setup_script.sh
   chmod +x pythonanywhere_setup_script.sh
   ```
3. **Run the script**:
   ```bash
   # Without GitHub repository:
   ./pythonanywhere_setup_script.sh
   
   # With GitHub repository:
   ./pythonanywhere_setup_script.sh https://github.com/yourusername/mood-tracker.git
   ```
4. **Follow the instructions** printed by the script

5. **Install requirements**:
   ```bash
   cd ~/mood-tracker
   source venv/bin/activate
   pip install -r requirements.txt
   ```

## Step-by-Step Manual Setup (If Script Doesn't Work)

### 1. Create the Project Directory
```bash
# In your PythonAnywhere bash console
mkdir -p ~/mood-tracker
cd ~/mood-tracker
```

### 2. Set Up a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Upload or Clone Your Code
Option 1: Clone from GitHub
```bash
git clone https://github.com/yourusername/mood-tracker.git .
```

Option 2: Upload via the Files tab in PythonAnywhere dashboard

### 4. Install Dependencies
```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('vader_lexicon')"
```

### 5. Configure Web App in PythonAnywhere

1. **Go to the Web tab** in your PythonAnywhere dashboard and click "Add a new web app"

2. **Choose your domain name** (it will be yourusername.pythonanywhere.com)

3. **Select "Manual configuration"** (not "Flask" or "Django")

4. **Choose Python 3.9** from the Python version dropdown

5. **Set your working directory:**
   - On the Web app configuration page, find the "Code" section
   - Set the "Source code" directory to `/home/yourusername/mood-tracker`
   - Set the "Working directory" to the same value

6. **Configure the WSGI file:**
   - Click on the WSGI configuration file link (usually named something like `/var/www/yourusername_pythonanywhere_com_wsgi.py`)
   - Replace all content with:

   ```python
   import sys
   import os
   
   # Add your project directory to the Python path
   path = '/home/yourusername/mood-tracker'
   if path not in sys.path:
       sys.path.append(path)
   
   # Point to your virtual environment
   activate_this = os.path.join(path, 'venv/bin/activate_this.py')
   with open(activate_this) as file_:
       exec(file_.read(), dict(__file__=activate_this))
   
   # Import your FastAPI app
   from app import app as application
   
   # ASGI application - needed for FastAPI
   application = application
   ```

7. **Set up the virtual environment:**
   - Under the "Virtualenv" section, click "Enter path manually"
   - Enter: `/home/yourusername/mood-tracker/venv`

### 6. Install FastAPI WSGI Adapter

FastAPI works with ASGI, but PythonAnywhere's free tier only supports WSGI, so we need an adapter:

1. **Go back to your Bash console** or open a new one
2. **Activate your virtual environment:**
   ```bash
   cd ~/mood-tracker
   source venv/bin/activate
   ```
3. **Install the adapter:**
   ```bash
   pip install uvicorn[standard] gunicorn
   pip install uvicorn-gunicorn-fastapi
   ```

4. **Create a WSGI adapter file:**
   ```bash
   cat > ~/mood-tracker/wsgi_adapter.py << 'EOL'
   from fastapi.middleware.wsgi import WSGIMiddleware
   from app import app
   from uvicorn.middleware.wsgi import WSGIMiddleware as UvicornWSGIMiddleware
   
   # Create a WSGI application from the FastAPI app
   application = WSGIMiddleware(app)
   EOL
   ```

5. **Update your WSGI file:**
   - Go back to the Web tab and edit your WSGI file again
   - Replace the last lines with:

   ```python
   # Import your WSGI adapter
   from wsgi_adapter import application
   ```

### 7. Static Files Configuration (Optional)

1. **Configure static files** if your app serves static content:
   - Go to the Web tab
   - Scroll down to "Static Files"
   - Add an entry:
     - URL: `/static/`
     - Directory: `/home/yourusername/mood-tracker/static`

### 8. Final Steps

1. **Reload your web app:**
   - Click the green "Reload" button on the Web tab

2. **Test your application:**
   - Visit your app at `yourusername.pythonanywhere.com`
   - Test the API endpoints at:
     - `yourusername.pythonanywhere.com/docs` (FastAPI Swagger UI)
     - `yourusername.pythonanywhere.com/predict` (your prediction endpoint)

3. **Check the error logs** if something doesn't work:
   - On the Web tab, click the "Error log" link

## Troubleshooting Common Issues

### If your app doesn't load:

1. **Check the error logs** in the Web tab
2. **Verify environment paths** are correct
3. **Check file permissions** - all files should be readable

### If dependencies are missing:

```bash
source ~/mood-tracker/venv/bin/activate
pip install -r ~/mood-tracker/requirements.txt
```

### If NLTK data is missing:

```bash
source ~/mood-tracker/venv/bin/activate
python -c "import nltk; nltk.download('vader_lexicon', download_dir='/home/yourusername/nltk_data')"
```

Then add this to the top of your app.py:

```python
import nltk
nltk.data.path.append('/home/yourusername/nltk_data')
```

## Limitations of Free Tier

- Your app goes to sleep after 3 months of inactivity
- CPU quota is limited (enough for development/demo)
- Memory is limited to 512MB
- Web apps are throttled when not used for a while
