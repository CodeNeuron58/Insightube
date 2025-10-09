import logging

def get_logger(name):
    # Create a logger instance
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:  # Prevent adding multiple handlers
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        ch.setFormatter(formatter)

        # Add the console handler
        logger.addHandler(ch)

    return logger
