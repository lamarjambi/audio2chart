"""
Audio (wav/mp3/ogg/flac) -> Clone Hero .chart  (auto-charter)

Usage:
    py audio_to_chart.py                         -> asks for audio file, output folder, difficulty
    py audio_to_chart.py song.mp3                -> ./song/notes.chart (all 4 difficulties)
    py audio_to_chart.py song.mp3 out_folder
    py audio_to_chart.py song.mp3 out_folder medium   -> only MediumSingle in the chart
                                             (easy / medium / hard / expert / all)

How it works:
  1. Beat tracking  -> tempo map (one B event per beat, so the chart follows tempo drift)
  2. Onset detection -> note times (+ onset strength)
  3. Snap onsets to the beat grid per difficulty, keep the strongest when crowded
  4. Pitch (chroma) of each note -> fret moves up/down with the melody
  5. Strong downbeats on Hard/Expert -> 2-note chords; long gaps -> sustains

Output works in Clone Hero / YARG / Moonscraper, and with chart_to_markers.py.
"""
import shutil
import sys
from pathlib import Path

import librosa
import numpy as np

# --- config ---
RESOLUTION = 192            # ticks per beat (Clone Hero standard)
SR = 22050
HOP = 256                   # analysis hop: 256/22050 = 11.6ms time resolution
DIFFICULTIES = {
    # name           grid (fraction of a beat)  keep (top % by strength)  frets  chords
    "ExpertSingle": dict(grid=1 / 4, keep=1.00, frets=5, chords=True),
    "HardSingle":   dict(grid=1 / 2, keep=0.80, frets=5, chords=True),
    "MediumSingle": dict(grid=1 / 2, keep=0.55, frets=4, chords=False),
    "EasySingle":   dict(grid=1,     keep=0.35, frets=3, chords=False),
}
CHORD_TOP_PCT = 0.10        # strongest 10% of notes that land on a beat become chords
SUSTAIN_MIN_BEATS = 1.0     # a note followed by a gap >= this many beats gets a sustain
STEADY_STD_SEC = 0.03      # beats within this of a straight line -> constant BPM
SMOOTH_BEATS = 8            # otherwise average tempo over this many beats
# --------------
DIFF_NAMES = {"easy": "EasySingle", "medium": "MediumSingle",
              "hard": "HardSingle", "expert": "ExpertSingle"}


def fail(msg):
    raise SystemExit(f"FAIL: {msg}")


def analyze(path):
    y, sr = librosa.load(path, sr=SR, mono=True)
    if len(y) < sr:
        fail("audio shorter than 1 second")
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=env, sr=sr, hop_length=HOP)
    beats = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP)
    if len(beats) < 4:
        fail(f"only {len(beats)} beats detected; is the file silent or arrhythmic?")

    on_frames = librosa.onset.onset_detect(onset_envelope=env, sr=sr, hop_length=HOP)
    # onset frames trail the attack by ~1 hop
    on_times = librosa.frames_to_time(on_frames, sr=sr, hop_length=HOP) - HOP / sr
    strength = env[on_frames]

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP)
    pitch = chroma[:, np.minimum(on_frames + 4, chroma.shape[1] - 1)].argmax(axis=0)

    return dict(duration=len(y) / sr, tempo=float(np.atleast_1d(tempo)[0]), beats=beats,
                on_times=on_times, strength=strength, pitch=pitch)


def beat_grid(beats, on_times, duration):
    """Beat times starting at t=0 (lead-in beat added) and extended past the song end."""
    # beat tracker jitters by ~1 frame (23ms); steady songs get a straight-line fit
    i = np.arange(len(beats))
    slope, start = np.polyfit(i, beats, 1)
    if np.std(beats - (start + slope * i)) < STEADY_STD_SEC:
        beats = start + slope * i
    else:  # tempo really drifts: smooth intervals instead
        ivls = np.convolve(np.diff(beats), np.ones(SMOOTH_BEATS) / SMOOTH_BEATS, mode="same")
        beats = beats[0] + np.concatenate([[0], np.cumsum(ivls)])
    ivl = float(np.median(np.diff(beats)))
    # beat tracker phase is off by ~1-2 frames; shift the grid so onsets sit on 1/16 lines
    sub = np.interp(on_times, beats, np.arange(len(beats))) * 4
    near = np.abs(sub - np.round(sub)) < 0.25
    if near.sum() >= 8:
        beats = beats + float(np.median((sub - np.round(sub))[near])) * ivl / 4
    first = beats[0] - np.floor(beats[0] / ivl) * ivl      # earliest beat >= 0 on the grid
    b = list(np.arange(first, beats[0] - ivl / 2, ivl)) + list(beats)
    if b[0] > 0.05:
        b.insert(0, 0.0)                                    # short lead-in beat
    while b[-1] < duration + ivl:
        b.append(b[-1] + ivl)
    return np.array(b)


def times_to_beatpos(t, grid):
    return np.interp(t, grid, np.arange(len(grid)))


def assign_frets(pitches, n_frets):
    """Walk the frets with the melody: pitch up -> fret up, pitch down -> fret down."""
    frets, f = [], n_frets // 2
    prev = None
    for p in pitches:
        if prev is not None:
            d = (int(p) - int(prev) + 6) % 12 - 6          # semitone jump, -6..5
            step = 0 if d == 0 else (1 if abs(d) < 5 else 2) * (1 if d > 0 else -1)
            f = min(max(f + step, 0), n_frets - 1)       # clamp at the edges
        frets.append(f)
        prev = p
    return frets


def make_difficulty(a, beatpos, cfg):
    step = cfg["grid"] * RESOLUTION
    ticks = np.round(beatpos * RESOLUTION / step) * step

    # keep strongest notes first, one per grid tick, until the keep-quota is met
    order = np.argsort(-a["strength"])
    quota = max(1, int(len(order) * cfg["keep"]))
    chosen = {}
    for i in order:
        tk = int(ticks[i])
        if tk not in chosen:
            chosen[tk] = i
            if len(chosen) >= quota:
                break

    tks = sorted(chosen)
    idx = [chosen[t] for t in tks]
    frets = assign_frets(a["pitch"][idx], cfg["frets"])
    strength = a["strength"][idx]
    chord_cut = np.quantile(a["strength"], 1 - CHORD_TOP_PCT)

    lines = []
    for k, (tk, f, s) in enumerate(zip(tks, frets, strength)):
        nxt = tks[k + 1] if k + 1 < len(tks) else tk + 4 * RESOLUTION
        gap = nxt - tk
        sus = int(gap - RESOLUTION / 4) if gap >= SUSTAIN_MIN_BEATS * RESOLUTION else 0
        notes = [f]
        if cfg["chords"] and s >= chord_cut and tk % RESOLUTION == 0:
            notes.append(f + 2 if f + 2 < cfg["frets"] else f - 2)
        for n in sorted(notes):
            lines.append(f"  {tk} = N {n} {sus}")
    return lines, len(tks)


def write_chart(path, name, diffs):
    a = analyze(path)
    full = beat_grid(a["beats"], a["on_times"], a["duration"])
    beatpos = times_to_beatpos(a["on_times"], full)

    sync, last = ["  0 = TS 4"], None
    for i, ivl in enumerate(np.diff(full)):
        bpm_x1000 = int(round(60_000 / ivl))               # .chart stores BPM * 1000
        if bpm_x1000 != last:
            sync.append(f"  {i * RESOLUTION} = B {bpm_x1000}")
            last = bpm_x1000

    out = [
        "[Song]", "{",
        f'  Name = "{name}"', '  Artist = "Unknown"', '  Charter = "audio_to_chart.py"',
        "  Offset = 0", f"  Resolution = {RESOLUTION}", '  Player2 = bass',
        "  Difficulty = 0", "  PreviewStart = 0", "  PreviewEnd = 0",
        '  Genre = "rock"', '  MediaType = "cd"', "}",
        "[SyncTrack]", "{", *sync, "}",
        "[Events]", "{", "}",
    ]
    stats = {}
    for diff in diffs:
        cfg = DIFFICULTIES[diff]
        lines, n = make_difficulty(a, beatpos, cfg)
        out += [f"[{diff}]", "{", *lines, "}"]
        stats[diff] = n
    return "\n".join(out) + "\n", a, stats


def parse_difficulty(text):
    t = (text or "all").strip().lower()
    if t == "all":
        return list(DIFFICULTIES)
    for key in DIFF_NAMES:
        if key.startswith(t):                      # "e", "med", "hard" all work
            return [DIFF_NAMES[key]]
    fail(f"unknown difficulty '{text}' (use easy / medium / hard / expert / all)")


def main(audio, out_dir=None, difficulty=None):
    diffs = parse_difficulty(difficulty)
    audio = Path(audio)
    if not audio.exists():
        fail(f"{audio} not found")
    out_dir = Path(out_dir or audio.stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    chart, a, stats = write_chart(str(audio), audio.stem, diffs)
    (out_dir / "notes.chart").write_text(chart, encoding="utf-8")
    (out_dir / "song.ini").write_text(
        f"[song]\nname = {audio.stem}\nartist = Unknown\ncharter = audio_to_chart.py\n"
        f"song_length = {int(a['duration'] * 1000)}\n", encoding="utf-8")
    shutil.copy(audio, out_dir / f"song{audio.suffix.lower()}")

    print(f"{audio.name}: {a['duration']:.1f}s | ~{a['tempo']:.0f} BPM | "
          f"{len(a['beats'])} beats | {len(a['on_times'])} onsets")
    for d, n in stats.items():
        print(f"  {d:<13} {n:4d} notes")
    print(f"-> {out_dir / 'notes.chart'}")


def ask(prompt):
    # strips quotes Windows adds when you drag a file into the terminal
    return input(prompt).strip().strip('"').strip("'")


if __name__ == "__main__":
    if len(sys.argv) > 4:
        fail("usage: py audio_to_chart.py [audio file] [out folder] [difficulty]")
    if len(sys.argv) >= 2:
        main(*sys.argv[1:])
    else:
        audio = ask("Audio file (type name or drag file here): ")
        if not audio:
            fail("no audio file given")
        out = ask(f"Output folder [Enter = {Path(audio).stem}]: ")
        diff = ask("Difficulty (easy / medium / hard / expert) [Enter = all]: ")
        main(audio, out or None, diff or None)