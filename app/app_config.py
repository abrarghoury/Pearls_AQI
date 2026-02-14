from config.settings import settings
APP_CONFIG = {
    "title": "Pearls AQI Predictor",
    "icon": "🌫️",
    "layout": "wide",
    "initial_sidebar_state": "collapsed",
}

CITY_CONFIG = {
    "city": settings.CITY,
    "timezone": settings.TIMEZONE
}
