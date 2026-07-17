from review_queue import get_review_jobs

jobs = get_review_jobs()

for job in jobs:
    print(job["filename"], job["status"])