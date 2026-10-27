"""Versionsnummer der App (MAJOR.MINOR.PATCH).

Bei jeder Weitergabe erhöhen. Was ein Update von einer älteren Version aus tun muss, steht in
setup_env.py (MIGRATIONS) und wird von update.bat / update.sh Schritt für Schritt ausgeführt.
"""
APP_VERSION = "1.1.0"
# Installationen ohne Versionsangabe in config.json stammen aus der Zeit vor der Versionierung.
UNVERSIONED = "1.0.0"
