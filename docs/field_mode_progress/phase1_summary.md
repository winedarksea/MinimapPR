# Phase 1 sync spike summary

## Gate

| item | status | models | runs |
|---|---|---|---|
| bird_band_leakage | Pass | pixel_10_pro_xl, pixel_7a, sm-a037f | pipeline_check desk_pipeline_check; pipeline_check desk_pipeline_check; A6 soak_1m; A6 soak_1m; A3 pixels_check; A3 pixels_check; A6 pixels_soak_1m; A6 pixels_soak_1m; A1 pixels_v18_b_right; A1 pixels_v18_b_right; A3 pixels_v18_left; A3 pixels_v18_left |
| capture_path | Info | pixel_10_pro_xl, pixel_7a, sm-a037f | pipeline_check desk_pipeline_check; pipeline_check desk_pipeline_check; A6 soak_1m; A6 soak_1m; A3 pixels_check; A3 pixels_check; A6 pixels_soak_1m; A6 pixels_soak_1m; A1 pixels_v18_b_right; A1 pixels_v18_b_right; A3 pixels_v18_left; A3 pixels_v18_left |
| clock_residual_50us_30min | Fail | pixel_10_pro_xl, pixel_7a, sm-a037f | pipeline_check desk_pipeline_check; A6 soak_1m; A6 pixels_soak_1m |
| locked_screen_playback | Fail | pixel_10_pro_xl, pixel_7a, sm-a037f | pipeline_check desk_pipeline_check; A6 soak_1m; A6 soak_1m; A3 pixels_check; A3 pixels_check; A6 pixels_soak_1m; A6 pixels_soak_1m |
| range_10cm_at_20m | Info | pixel_10_pro_xl, pixel_7a, sm-a037f | A6 soak_1m; A3 pixels_check; A6 pixels_soak_1m; A1 pixels_v18_b_right; A3 pixels_v18_left |
| self_onset_stable_10us | Fail | pixel_7a, sm-a037f | pipeline_check desk_pipeline_check; pipeline_check desk_pipeline_check |
| self_path_sum_calibration | Info | pixel_10_pro_xl, pixel_7a, sm-a037f | A6 soak_1m; A3 pixels_check; A6 pixels_soak_1m; A1 pixels_v18_b_right; A3 pixels_v18_left |
| speaker_channel_map | Info | pixel_10_pro_xl, pixel_7a, sm-a037f | pipeline_check desk_pipeline_check; pipeline_check desk_pipeline_check; A6 soak_1m; A6 soak_1m; A3 pixels_check; A3 pixels_check; A6 pixels_soak_1m; A6 pixels_soak_1m; A1 pixels_v18_b_right; A1 pixels_v18_b_right; A3 pixels_v18_left; A3 pixels_v18_left |

## Range vs band and distance

| run | waveform | band Hz | tape m | mode | median m | error mm | robust σ mm | rounds | remote SNR A/B dB |
|---|---|---|---|---|---|---|---|---|---|
| pipeline_check desk_pipeline_check | updown_hfm | 13000–16000 | — | none | 0.6760 | — | 43.4 | 47 | 20/25 |
| pipeline_check desk_pipeline_check | updown_hfm | 13000–16000 | — | tracked_rate_ratio | 0.6704 | — | 43.4 | 47 | 20/25 |
| pipeline_check desk_pipeline_check | updown_hfm | 13000–16000 | — | double_sided | 0.6698 | — | 39.0 | 47 | 20/25 |
| A6 soak_1m | updown_hfm | 13000–16000 | 1.00 | none | 1.1517 | 151.7 | 1.7 | 291 | 34/38 |
| A6 soak_1m | updown_hfm | 13000–16000 | 1.00 | tracked_rate_ratio | 1.1472 | 147.2 | 1.6 | 291 | 34/38 |
| A6 soak_1m | updown_hfm | 13000–16000 | 1.00 | double_sided | 1.1473 | 147.3 | 1.7 | 291 | 34/38 |
| A3 pixels_check | updown_hfm | 13000–16000 | 1.00 | none | 0.9507 | -49.3 | 6.1 | 37 | 34/26 |
| A3 pixels_check | updown_hfm | 13000–16000 | 1.00 | tracked_rate_ratio | 0.9468 | -53.2 | 5.7 | 37 | 34/26 |
| A3 pixels_check | updown_hfm | 13000–16000 | 1.00 | double_sided | 0.9480 | -52.0 | 8.8 | 37 | 34/26 |
| A6 pixels_soak_1m | updown_hfm | 13000–16000 | 1.00 | none | 0.9891 | -10.9 | 0.7 | 457 | 38/27 |
| A6 pixels_soak_1m | updown_hfm | 13000–16000 | 1.00 | tracked_rate_ratio | 0.9860 | -14.0 | 0.7 | 457 | 38/27 |
| A6 pixels_soak_1m | updown_hfm | 13000–16000 | 1.00 | double_sided | 0.9861 | -13.9 | 0.6 | 457 | 38/27 |
| A1 pixels_v18_b_right | updown_hfm | 13000–16000 | 1.00 | none | 0.9891 | -10.9 | 0.2 | 44 | 38/27 |
| A1 pixels_v18_b_right | updown_hfm | 13000–16000 | 1.00 | tracked_rate_ratio | 0.9863 | -13.7 | 0.2 | 44 | 38/27 |
| A1 pixels_v18_b_right | updown_hfm | 13000–16000 | 1.00 | double_sided | 0.9863 | -13.7 | 0.1 | 44 | 38/27 |
| A3 pixels_v18_left | updown_hfm | 13000–16000 | 1.00 | none | 0.9145 | -85.5 | 11.3 | 42 | 35/26 |
| A3 pixels_v18_left | updown_hfm | 13000–16000 | 1.00 | tracked_rate_ratio | 0.9113 | -88.7 | 11.3 | 42 | 35/26 |
| A3 pixels_v18_left | updown_hfm | 13000–16000 | 1.00 | double_sided | 0.9126 | -87.4 | 11.6 | 42 | 35/26 |

## Speaker channel map

| run | emitter | channel | orientation | listener | median SNR dB | heard |
|---|---|---|---|---|---|---|
| pipeline_check desk_pipeline_check | pixel_7a | right | FlatFaceUp | sm-a037f | 24.9 | 48/48 |
| pipeline_check desk_pipeline_check | sm-a037f | right | Unknown | pixel_7a | 19.8 | 48/48 |
| A6 soak_1m | pixel_7a | right | FlatFaceUp | sm-a037f | 38.1 | 464/464 |
| A6 soak_1m | sm-a037f | right | Unknown | pixel_7a | 34.4 | 460/465 |
| A3 pixels_check | pixel_7a | left | FlatFaceUp | pixel_10_pro_xl | 25.9 | 40/40 |
| A3 pixels_check | pixel_10_pro_xl | left | FlatFaceUp | pixel_7a | 33.6 | 40/40 |
| A6 pixels_soak_1m | pixel_7a | left | FlatFaceUp | pixel_10_pro_xl | 26.6 | 458/458 |
| A6 pixels_soak_1m | pixel_10_pro_xl | right | FlatFaceUp | pixel_7a | 37.8 | 458/458 |
| A1 pixels_v18_b_right | pixel_7a | left | FlatFaceUp | pixel_10_pro_xl | 26.6 | 45/45 |
| A1 pixels_v18_b_right | pixel_10_pro_xl | right | FlatFaceUp | pixel_7a | 37.7 | 45/45 |
| A3 pixels_v18_left | pixel_7a | left | FlatFaceUp | pixel_10_pro_xl | 25.7 | 43/43 |
| A3 pixels_v18_left | pixel_10_pro_xl | left | FlatFaceUp | pixel_7a | 34.5 | 43/43 |
