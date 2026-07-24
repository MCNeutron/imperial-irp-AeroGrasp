import time
from datetime import datetime

counter = 0

while True:
    print(f"{counter}: {datetime.now()}", flush=True)
    counter += 1
    time.sleep(10)