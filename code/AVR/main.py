"""
Main module for AVR.

Author: Lucy Roellecke
Years: 2023
"""

# %% Import
import logging

from AVR.configs import config, params, paths
from AVR.preprocessing.freesurfer import foo

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
logger = logging.getLogger(__name__)


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def main():
    """Run the main."""
    print(f"Access service x using my private key: {config.service_x.api_key}")
    foo()
    print(f"{paths.results.GLM}/{params.weight_decay}/")
    logger.info("My first log entry. Use pre-configured loggers! You can change logging.configs in 'code/configs/config.toml'")


# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

if __name__ == "__main__":
    # Run main
    main()

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
