from pathlib import Path


THUMBNAIL_FOLDER = Path("static/thumbnails")


def get_gallery():

    videos = []

    if not THUMBNAIL_FOLDER.exists():
        return videos

    for image in sorted(THUMBNAIL_FOLDER.glob("*.jpg")):

        videos.append({
            "name": image.stem,
            "thumbnail": image.name
        })

    return videos