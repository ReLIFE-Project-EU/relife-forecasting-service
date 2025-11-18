from importlib.metadata import version

from fastapi import FastAPI

from relife_forecasting.config.logging import configure_logging
from relife_forecasting.routes import auth, examples, health

# Dynamically determine the package name
package_name = __name__.split(".")[0]

# Get version dynamically
package_dist_name = package_name.replace("_", "-")

try:
    __version__ = version(package_dist_name)
except ImportError:
    __version__ = "development"

configure_logging()

app = FastAPI(
    title="ReLIFE Forecasting Service",
    description="ReLIFE service for building energy forecasting",
    version=__version__,
)

app.include_router(health.router)
app.include_router(auth.router)
app.include_router(examples.router)
