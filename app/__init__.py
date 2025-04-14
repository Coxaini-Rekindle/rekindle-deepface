import os

from flask import Flask, jsonify
from flask_cors import CORS

from app.routes.api import api_bp


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    CORS(app)

    # Register blueprints
    app.register_blueprint(api_bp)

    # Health check endpoint
    @app.route("/healthcheck", methods=["GET"])
    def healthcheck():
        """Simple health check endpoint."""
        return jsonify(
            {"status": "ok", "message": "Face recognition service is running"}
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
