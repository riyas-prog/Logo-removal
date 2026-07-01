# Watermark Removal — Team Tool

A small web app for your team: upload videos (one or many at once), and it
automatically finds and removes a fixed-position watermark/logo, no manual
cropping needed.

## What it does

For each uploaded video, it tries two detection methods automatically and
uses whichever works:

1. **Fast scan** (`static_variance`) — finds a logo that sits in one spot
   and doesn't move or animate (most app/screen-recorder watermarks).
2. **Deep scan** (`bootstrapped_template`) — if the fast scan finds
   nothing, this tries candidate regions around the frame's corners/edges
   and checks which one matches consistently across many frames. This
   catches watermarks that sit in a fixed spot but have their own
   animation/glow (like an animated ad badge), which would otherwise fool
   the fast scan.

If neither method finds a confident match, the video is marked **failed**
with an explanation — it does NOT guess and risk damaging the video.

This only handles **fixed-position** watermarks (same spot for the whole
video). A watermark that visibly moves/slides around during playback isn't
covered by this version — see "Limitations" below.

## Local setup (try it yourself first)

```bash
pip install -r requirements.txt --break-system-packages
export TEAM_TOOL_PASSWORD="pick-something-only-your-team-knows"
python3 app.py
```
Open `http://localhost:5000`, log in with that password, and try uploading
a video.

## Sharing this with your team

You have a few options, roughly easiest-to-hardest:

### Option A — Run it on your own machine, share over your network
Quick for a small team on the same office/VPN network. Run the steps above,
then find your machine's local IP (`ipconfig`/`ifconfig`) and share
`http://<your-ip>:5000` with teammates. Only works while your machine is on
and the script is running.

### Option B — Deploy to a small cloud host (recommended for real use)
Services like Render, Railway, Fly.io, or a small DigitalOcean/AWS instance
can run this continuously. General steps:
1. Push this folder to a private GitHub repo.
2. Connect that repo to your hosting service of choice.
3. Set the `TEAM_TOOL_PASSWORD` environment variable in the host's
   dashboard (never commit it to the repo).
4. Set the start command to `python3 app.py` (the app already reads `PORT`
   from the environment, which most hosts set automatically).
5. Share the resulting URL with your team.

Most of these have a free or low-cost tier sufficient for light internal
use. Video processing is CPU-bound and can be slow on the smallest tiers —
if your team uploads a lot of long videos, a host with more CPU will feel
much more responsive.

### Option C — Run on a server you already have
If your team already has an internal server, copy this folder there, set
the environment variable, and run it (ideally under a process manager like
`systemd` or `pm2` so it restarts if it crashes, and behind a reverse proxy
like nginx if you want a real domain/HTTPS).

## Important notes before sharing

- **Change the password.** The default is `changeme` — set
  `TEAM_TOOL_PASSWORD` to something real before anyone else uses this.
- **This is for your team's own content.** It's built and password-gated
  for internal use processing your own footage/ad creative — not intended
  as a public tool.
- **Disk space**: uploaded and processed videos are stored in `uploads/`
  and `outputs/`. Nothing deletes them automatically yet. For regular use,
  add a cleanup step (a cron job clearing files older than a day, or call
  the existing `/cleanup/<job_id>` endpoint after a teammate downloads
  their result) so disk usage doesn't grow unbounded.
- **Concurrent jobs**: jobs run in background threads in a single process.
  Fine for a small team uploading a handful of videos at a time. If your
  team is large or uploads dozens of videos simultaneously, consider a
  real task queue (Celery/RQ + Redis) instead — ask me if you hit that
  point and I can help upgrade it.
- **Job history isn't persistent.** Job status lives in memory; if the app
  restarts, in-progress/finished job records are lost (downloaded files in
  `outputs/` are unaffected unless cleaned up). For a small team doing
  occasional batches this is a non-issue; flag it if you need durability.

## Limitations (be upfront with your team about these)

- Works on **fixed-position** watermarks only (same corner/spot for the
  whole video). A logo that visibly slides or moves during playback needs
  the original manual-crop + per-frame-matching mode instead (the script
  this app is built on supports it — `core_processing.py` → `logo_path`
  mode — just not wired into this particular UI yet).
- Removal quality depends on what's behind the watermark. Inpainting fills
  the region using nearby pixels in the same frame — it works well on busy
  textures (crowds, grass, etc., as tested) and less well on very flat,
  sharp-edged backgrounds where a smudge may be more visible.
- If detection fails (job shows "Failed"), it means neither method found a
  confident match — usually because the watermark genuinely isn't
  fixed-position, or it's very small/subtle. Don't force it; check the
  source video.

## Files
- `app.py` — Flask web app (routes, upload handling, job tracking)
- `core_processing.py` — video processing engine (frame I/O, masking,
  inpainting, audio muxing)
- `auto_detect.py` — fast variance-based static-region detector
- `auto_locked_detect.py` — fallback bootstrapped-template detector for
  animated-but-fixed-position watermarks
- `templates/` — login and main upload page
