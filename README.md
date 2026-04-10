# braille-scope

A tiny terminal “waterfall” / spectrogram viewer that renders frequency intensity using **Unicode Braille** characters.

Instead of drawing pixels or colors, each output character is a Braille cell (U+2800..U+28FF). A Braille cell has up to 8 dots; this program uses the dots as a compact vertical bargraph:
- more dots lit = higher magnitude (louder) in that frequency bin
- each printed line is one time slice; lines scroll downward to form the waterfall

It can read from an audio file (via libsndfile) or from a live JACK input.

## Build

Dependencies:
- CMake ≥ 3.25
- FFTW3 (single-precision, `fftw3f`)
- JACK
- libsndfile
- Boost headers (for `lockfree`)

```sh
cmake -S . -B build
cmake --build build -j
```

## Run

### From a file
```sh
./build/braille-scope path/to/audio.wav
```

### Live with JACK
```sh
./build/braille-scope --jack
```

## Options

Common useful knobs:
- `--fps N` frames per second (default `30`)
- `--width N` output width in braille glyphs (default `79`)
- `--min-freq HZ` / `--max-freq HZ` frequency range (or musical note like c0)
- `--min-dbfs DB` / `--max-dbfs DB` display range in dBFS
- `--linear` use linear frequency mapping
- `--bins-per-octave N` or `--cents-per-bin CENTS` for log mapping
- `--stereo` mirror 2 channels around the center axis (requires exactly 2 channels)

Full help:
```sh
./build/braille-scope --help
```
