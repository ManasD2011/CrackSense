"""
CrackSense – Flask Application Entry Point
"""

import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask
from flask_cors import CORS

from api.routes.predict import predict_bp


def create_app() -> Flask:
    app = Flask(
        __name__,
        static_folder=os.path.join(os.path.dirname(__file__), '..', 'frontend'),
        static_url_path='/static',
    )

    # ── Config ────────────────────────────────────────────────────────────────
    from src.utils.config import MAX_CONTENT_MB
    app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_MB * 1024 * 1024

    # ── CORS ──────────────────────────────────────────────────────────────────
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # ── Blueprints ────────────────────────────────────────────────────────────
    app.register_blueprint(predict_bp, url_prefix='/api')

    # ── Serve frontend ────────────────────────────────────────────────────────
    from flask import send_from_directory

    @app.route('/', defaults={'path': ''})
    @app.route('/<path:path>')
    def serve_frontend(path):
        frontend_dir = os.path.join(
            os.path.dirname(__file__), '..', 'frontend'
        )
        if path and os.path.exists(os.path.join(frontend_dir, path)):
            return send_from_directory(frontend_dir, path)
        return send_from_directory(frontend_dir, 'index.html')

    # ── Health check ──────────────────────────────────────────────────────────
    @app.route('/api/health')
    def health():
        return {"status": "ok", "service": "CrackSense API"}, 200

    return app


if __name__ == '__main__':
    from src.utils.config import FLASK_HOST, FLASK_PORT, DEBUG_MODE
    app = create_app()
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=DEBUG_MODE)
