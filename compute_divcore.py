import sys
import divprop.logging as logging

from ciphers import ciphers


logging.setup()
log = logging.getLogger(__name__)

name = sys.argv[1]
cipher = ciphers[name]()

logging.addFileHandler(f"logs/divcore.{name}")
log.info(f"cipher {name} {cipher}")

snd = cipher.make_sandwich()
snd.compute_divcore(filename=f"data/{name}.divcore")
