#!/usr/bin/env python3
"""Capture the offline demo and record synthetic narration and captions on macOS."""

import argparse
import json
import math
import subprocess
from pathlib import Path
from textwrap import wrap

CAPTURE = r"""
const fs = require('node:fs');
const path = require('node:path');
const {pathToFileURL} = require('node:url');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
async function main() {
  const directory = process.argv[1];
  const frames = JSON.parse(fs.readFileSync(path.join(directory, 'frames.json')));
  const url = pathToFileURL(path.join(directory, 'midpoint.html'));
  url.search = 'capture';
  const browser = await chromium.launch();
  try {
    const page = await browser.newPage({viewport: {width: 1280, height: 720}, deviceScaleFactor: 1.5});
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(url.href);
    for (let i = 0; i < frames.length; i++) {
      const hash = frames[i].hash;
      await page.evaluate(value => { location.hash = value; }, hash);
      await page.waitForFunction(value => {
        const [slide, step] = value.slice(1).split('-').map(Number);
        const scene = document.querySelectorAll('.scene')[slide];
        return !scene.hidden && [...scene.querySelectorAll('[data-reveal]')]
          .every(el => el.hidden === (Number(el.dataset.reveal) > step));
      }, hash);
      const fits = await page.locator('.scene:visible').evaluate(scene => {
        const footer = scene.querySelector('footer').getBoundingClientRect();
        const content = [...scene.children].filter(el => el.tagName !== 'FOOTER');
        return content.every(el => el.getBoundingClientRect().bottom + 12 < footer.top)
          && scene.scrollWidth <= scene.clientWidth;
      });
      if (!fits) throw new Error('Slide overlaps footer or overflows at ' + hash);
      await page.screenshot({path: path.join(directory, 'frame-' + i + '.png')});
    }
    if (errors.length) throw new Error(errors.join(', '));
  } finally {
    await browser.close();
  }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
"""


def run(*args):
    subprocess.run(args, check=True)


def duration(path):
    return float(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ]))


def timestamp(seconds):
    millis = round(seconds * 1000)
    minutes, tail = divmod(millis, 60000)
    hours, minutes = divmod(minutes, 60)
    seconds, millis = divmod(tail, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"


def captions_for(text, start, length, first_number):
    chunks = wrap(text, width=77)
    weight = sum(map(len, chunks))
    cues = []
    for index, chunk in enumerate(chunks):
        end = start + length * len(chunk) / weight
        cues.append(f"{first_number + index}\n{timestamp(start)} --> {timestamp(end)}\n"
                    + "\n".join(wrap(chunk, width=41)))
        start = end
    return cues


def render(build):
    frames = json.loads((build / "frames.json").read_text())
    run("node", "-e", CAPTURE, str(build))
    clips, captions = [], []
    elapsed = 0
    for index, frame in enumerate(frames):
        prefix = build / f"frame-{index}"
        audio, clip = prefix.with_suffix(".aiff"), prefix.with_suffix(".mp4")
        run("say", "-v", "Samantha", "-r", "166", "-o", str(audio), frame["narration"])
        spoken = duration(audio)
        length = math.ceil((spoken + 0.8) * 30) / 30
        run("ffmpeg", "-v", "error", "-y", "-loop", "1", "-framerate", "30",
            "-i", str(prefix.with_suffix(".png")), "-i", str(audio),
            "-af", "adelay=150:all=1,apad", "-t", str(length), "-pix_fmt", "yuv420p",
            "-c:v", "libx264", "-preset", "veryfast", "-tune", "stillimage", "-c:a", "aac",
            "-ar", "48000", "-ac", "2", str(clip))
        captions.extend(captions_for(frame["narration"], elapsed + 0.15, spoken, len(captions) + 1))
        elapsed += length
        clips.append(f"file '{clip.name}'")
    playlist = build / "playlist.txt"
    playlist.write_text("\n".join(clips) + "\n")
    srt = build / "midpoint-demo.srt"
    srt.write_text("\n\n".join(captions) + "\n")
    run("ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "1", "-i", str(playlist),
        "-i", str(srt), "-map", "0:v", "-map", "0:a", "-map", "1:0", "-c:v", "copy",
        "-c:a", "copy", "-c:s", "mov_text", "-metadata:s:s:0", "language=eng",
        "-movflags", "+faststart", str(build / "midpoint-demo.mp4"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    render(parser.parse_args().build.resolve())
