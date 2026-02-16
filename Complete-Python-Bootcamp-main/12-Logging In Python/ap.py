import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s-%(name)s-%(levelname)s-%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("ap.log"),
        logging.StreamHandler()
    ]
)
logger=logging.getLogger('halua')

def greet(name):
    logger.debug(f'Greeting {name}')
    return f'hyy {name}'


greet('Gajar')

