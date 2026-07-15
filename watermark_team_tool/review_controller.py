from review_queue import get_review_jobs


def get_review_session():

    jobs = get_review_jobs()

    if not jobs:
        return None

    return {
        "current": 0,
        "total": len(jobs),
        "jobs": jobs
    }