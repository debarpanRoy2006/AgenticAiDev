import os
import datetime # Added for timestamp in memory

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SECRET_KEY = 'your-secret-key-here' # IMPORTANT: Change this in production!

DEBUG = True # Set to False in production

ALLOWED_HOSTS = [] # Add your domain names in production, e.g., ['yourdomain.com', 'www.yourdomain.com']

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'corsheaders', # Add corsheaders
    'agent_app',  # Your Django app
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware', # Add CORS middleware at the top
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'agent_ai_project.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(BASE_DIR, 'templates')], # Ensure this points to your templates folder
            'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'agent_ai_project.wsgi.application'

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_L10N = True

USE_TZ = True

STATIC_URL = '/static/'

# CORS settings
CORS_ALLOW_ALL_ORIGINS = True # For development, allows all origins. Be more restrictive in production.
# CORS_ALLOWED_ORIGINS = [
#     "http://127.0.0.1:5500", # Example if you're using Live Server in VS Code
#     "http://localhost:5500",
# ]7.0.0.1:5500)", # Example if you're using Live Server in VS Code
#     "http://localhost:5500",
# ]

# Gemini API Key (IMPORTANT: For production, use environment variables or a secure secret manager)
# For this demo, we'll leave it empty as per instructions.

GEMINI_API_KEY = "AIzaSyBn_5T6CZogy1yTmOfGqlIz4uigBMF7i8Q"
