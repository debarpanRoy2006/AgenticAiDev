import os
import dj_database_url
from .settings import * # Import all settings from your base settings.py

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = False

# Load environment variables from .env file (for local development)
# In Render, these will be set directly in the environment.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass # python-dotenv not installed, or not needed in production

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
# BASE_DIR is already defined in your base settings.py

# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/5.0/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
# Use an environment variable for SECRET_KEY in production
SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY', 'a-very-secret-key-for-local-dev-only')

# Allowed Hosts for Render deployment
# Render provides a primary external URL (e.g., your-app-name.onrender.com)
# and potentially other URLs if you configure custom domains.
ALLOWED_HOSTS = [os.environ.get('RENDER_EXTERNAL_HOSTNAME')]
# If you have custom domains, add them here:
# ALLOWED_HOSTS += ['yourcustomdomain.com', 'www.yourcustomdomain.com']


# Database configuration for Render (PostgreSQL)
# Render automatically provides a DATABASE_URL environment variable for PostgreSQL.
DATABASES = {
    'default': dj_database_url.config(
        default=os.environ.get('DATABASE_URL', 'sqlite:///db.sqlite3'), # Fallback to SQLite for local dev
        conn_max_age=600
    )
}

# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/5.0/howto/static-files/

# This is the directory where collectstatic will put all your static files
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')
STATIC_URL = '/static/'

# Enable WhiteNoise to serve static files efficiently in production
# Make sure to install WhiteNoise: pip install whitenoise
MIDDLEWARE.insert(1, 'whitenoise.middleware.WhiteNoiseMiddleware')
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'

# CORS settings for production
# Ensure your frontend URL is allowed in production
# Render frontend might be on a different domain
CORS_ALLOWED_ORIGINS = [
    os.environ.get('RENDER_FRONTEND_URL', 'http://localhost:8000'), # Default to localhost for safety
    # Add your actual frontend URL(s) here if they are not dynamically set by Render
    # e.g., "https://your-frontend-app.onrender.com",
]
# If you need to allow all origins in production (less secure, but sometimes used for APIs)
# CORS_ALLOW_ALL_ORIGINS = True # Only if necessary and understood risks

# Gemini API Key for production
# Render environment variable for Gemini API Key
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', 'AIzaSyBn_5T6CZogy1yTmOfGqlIz4uigBMF7i8Q') # Empty string fallback, should be set in Render


# Logging configuration (optional, but good for production)
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {process:d} {thread:d} {message}',
            'style': '{',
        },
        'simple': {
            'format': '{levelname} {message}',
            'style': '{',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'simple',
        },
    },
    'root': {
        'handlers': ['console'],
        'level': 'INFO',
    },
    'loggers': {
        'django': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': False,
        },
        'agent_app': { # Logger for your specific app
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': False,
        },
    },
}
