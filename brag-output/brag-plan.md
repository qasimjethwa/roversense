# brag-plan — RoverSense launch video

## Angle
A silent-failure horror hook for a technical audience: rover sensors don't break loudly, they lie quietly — and here's the ML system that catches them. Proof-first: real metrics, real model, real browser demo.

## Hook (first 2s)
"YOUR ROVER'S SENSORS ARE LYING TO YOU." — brutalist kinetic type slam, the word LYING glitches red.

## Highlights
1. **Train in Python** — IsolationForest, 64 trees, 18 context features, scikit-learn. Animated tree-splitting viz.
2. **Every tree → the browser** — the full forest exported to JSON, running client-side, bit-identical to sklearn (max diff 1.1e-16). No backend.
3. **Catches what matters** — five fault episodes with real detection times: pH freeze 21s, gas drift 7s, EMI spikes 17s, dropout 0s, gas leak 3s. Precision 0.986, F1 0.855.

## Punchline / outro
"ROVERSENSE — code + live demo at github.com/qasimjethwa/roversense". Built in a day.

## Tone
Punchy, technical, confident. No hype adjectives — the numbers are the swagger.

## Visual identity (merged styles)
1. **Dark glassmorphism mission-control** — the demo's own UI (glass cards, glowing sparklines, score gauge) is the primary visual material; the product in motion, not slides about it.
2. **Brutalist kinetic type** — huge condensed black-weight headlines slamming in with overshoot easing; mono labels for data.
3. **Aurora glow** — drifting cyan/violet/pink radial glows over near-black, dot-grid texture, vignette.
(Cosmos.so/explore/ui-ux was attempted for reference but is a JS-walled gallery — text fetch returned titles only, no visuals. The three styles above are the approved sample look Qasim rated "perfect", merged with the demo's real UI.)

## Storyboard (20s, 30fps)
| t | scene | content |
|---|---|---|
| 0.0–2.5 | hook | "YOUR ROVER'S SENSORS" / "ARE LYING TO YOU." (LYING glitches red); sub "a fouled probe fails silently" |
| 2.5–5.5 | reveal | ROVERSENSE wordmark, "ML anomaly detection for rover telemetry", pill "ISOLATION FOREST · 64 TREES" |
| 5.5–9.5 | train | animated isolation-tree splitting viz + "TRAIN IN PYTHON" / "64 trees · 18 features · scikit-learn" |
| 9.5–13.0 | export | "EVERY TREE → THE BROWSER" — JSON chip flies into a browser mock running the demo gauge; "bit-identical · max diff 1.1e-16" / "no backend" |
| 13.0–17.0 | catch | "CATCHES WHAT MATTERS" — 5 fault chips land one by one with detection times; counters: precision 0.986, F1 0.855 |
| 17.0–20.0 | outro | ROVERSENSE + pills github.com/qasimjethwa/roversense and live demo URL; "built in a day" |

## Audio
Synthesized electronic bed (chord pad Am–F–C–G, 122bpm arp, kick/hat, whooshes on cuts) + spoken voiceover (tts), music ducked −14dB under voice.

## Voiceover script (voiceover.txt, 49 words)
"Your rover's sensors are lying to you. RoverSense trains an isolation forest on rover telemetry in Python — then exports every single tree to run live in the browser. No backend. It catches frozen probes, drifting sensors, even gas leaks — in seconds. Code and live demo at the link below."
