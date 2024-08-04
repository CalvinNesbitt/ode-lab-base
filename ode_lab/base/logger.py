import logging

logger = logging.getLogger("ode-lab-base")
logger.setLevel(logging.DEBUG)


# Set up logging to console
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)  # Capture info and above levels in the console

# Create formatter and add it to the handlers
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(ch)
