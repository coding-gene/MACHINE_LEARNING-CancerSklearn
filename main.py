import logging
import time

start_time = time.time()
logging.basicConfig(filename='logs.txt',
                    filemode='a',
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')
logging.info('Početak izvršavanja zadatka.')
# noinspection PyBroadException
try:
    pass

except Exception:
    logging.exception('An error occurred during job performing:')
else:
    logging.info('Job ended.')
finally:
    #instagram.close_driver()
    logging.info(
        f'Job duration: {time.strftime("%H hours, %M minutes, %S seconds.", time.gmtime(time.time() - start_time))}\n')