#![windows_subsystem = "windows"]
use anyhow::Context;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use glium::{Display, Surface};
use imgui::Context as ImguiContext;
use imgui_glium_renderer::Renderer;
use imgui_winit_support::{HiDpiMode, WinitPlatform};
use rand::Rng;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

// Parameters shared between UI and Audio thread
struct AudioParams {
    frequency: f32,
    volume: f32,
    harmonic_3: f32,
    harmonic_5: f32,
    harmonic_7: f32,
    mechanical_hum: f32, // 100Hz
    metallic_res: f32,   // Ring mod / resonance
    noise: f32,
    saturation: f32,
    reverb: f32,
    is_playing: bool,
}

impl Default for AudioParams {
    fn default() -> Self {
        Self {
            frequency: 50.0,
            volume: 0.6,
            harmonic_3: 0.3,
            harmonic_5: 0.15,
            harmonic_7: 0.05,
            mechanical_hum: 0.4,
            metallic_res: 0.2,
            noise: 0.05,
            saturation: 0.6,
            reverb: 0.3,
            is_playing: true,
        }
    }
}

struct ReverbState {
    buffer_l: Vec<f32>,
    buffer_r: Vec<f32>,
    head_l: usize,
    head_r: usize,
}

impl ReverbState {
    fn new(sample_rate: f32) -> Self {
        // Simulate two coils / reflections with different path lengths
        // Left: ~20ms, Right: ~30ms
        let size_l = (sample_rate * 0.020) as usize;
        let size_r = (sample_rate * 0.030) as usize;
        Self {
            buffer_l: vec![0.0; size_l],
            buffer_r: vec![0.0; size_r],
            head_l: 0,
            head_r: 0,
        }
    }

    fn process(&mut self, input: f32, mix: f32) -> (f32, f32) {
        if mix <= 0.0 {
            return (input, input);
        }

        // Feedback for "box reflections"
        let feedback = 0.6;

        // Left Channel
        let len_l = self.buffer_l.len();
        let delayed_l = self.buffer_l[self.head_l];
        let new_l = input + delayed_l * feedback;
        self.buffer_l[self.head_l] = new_l.tanh(); // Soft clip feedback
        self.head_l = (self.head_l + 1) % len_l;

        // Right Channel
        let len_r = self.buffer_r.len();
        let delayed_r = self.buffer_r[self.head_r];
        let new_r = input + delayed_r * feedback;
        self.buffer_r[self.head_r] = new_r.tanh();
        self.head_r = (self.head_r + 1) % len_r;

        // Mix dry + wet
        let out_l = input + delayed_l * mix;
        let out_r = input + delayed_r * mix;

        (out_l, out_r)
    }
}

fn main() -> anyhow::Result<()> {
    // 1. Setup Audio
    let params = Arc::new(Mutex::new(AudioParams::default()));
    let audio_params = params.clone();

    // Keep the stream alive
    let _stream = setup_audio(audio_params)?;

    // 2. Setup Window & ImGui
    let event_loop = EventLoop::new();
    let context = glium::glutin::ContextBuilder::new().with_vsync(true);
    let builder = WindowBuilder::new()
        .with_title("Transformer Hum Simulator")
        .with_inner_size(winit::dpi::LogicalSize::new(1040.0, 600.0));

    let display =
        Display::new(builder, context, &event_loop).expect("Failed to initialize display");

    let mut imgui = ImguiContext::create();
    imgui.set_ini_filename(None);

    // Load Arial for Cyrillic support
    if let Ok(font_data) = std::fs::read("C:\\Windows\\Fonts\\arial.ttf") {
        imgui.fonts().add_font(&[
            imgui::FontSource::TtfData {
                data: &font_data,
                size_pixels: 20.0,
                config: Some(imgui::FontConfig {
                    rasterizer_multiply: 1.3,
                    glyph_ranges: imgui::FontGlyphRanges::cyrillic(),
                    ..imgui::FontConfig::default()
                }),
            }
        ]);
    }

    let mut platform = WinitPlatform::init(&mut imgui);
    {
        let gl_window = display.gl_window();
        let window = gl_window.window();
        platform.attach_window(imgui.io_mut(), window, HiDpiMode::Default);
    }

    let mut renderer = Renderer::init(&mut imgui, &display).expect("Failed to initialize renderer");

    let mut last_frame = Instant::now();

    event_loop.run(move |event, _, control_flow| {
        {
            let gl_window = display.gl_window();
            platform.handle_event(imgui.io_mut(), gl_window.window(), &event);
        }

        match event {
            Event::NewEvents(_) => {
                let now = Instant::now();
                imgui.io_mut().update_delta_time(now - last_frame);
                last_frame = now;
            }
            Event::MainEventsCleared => {
                let gl_window = display.gl_window();
                platform
                    .prepare_frame(imgui.io_mut(), gl_window.window())
                    .expect("Failed to prepare frame");
                gl_window.window().request_redraw();
            }
            Event::RedrawRequested(_) => {
                let ui = imgui.frame();
                let display_size = ui.io().display_size;
                let w = display_size[0];
                let h = display_size[1];

                // --- UI Logic ---
                {
                    let mut p = params.lock().unwrap();

                    ui.window("Transformer Settings")
                        .position([0.0, 0.0], imgui::Condition::Always)
                        .size([w * 0.60, h], imgui::Condition::Always)
                        .resizable(false)
                        .movable(false)
                        .collapsible(false)
                        .title_bar(true)
                        .build(|| {
                            ui.text("Генератор гула трансформаторной будки");
                            ui.separator();

                            ui.checkbox("Включить гул (Power On)", &mut p.is_playing);
                            ui.slider("Frequency (Hz)", 30.0, 80.0, &mut p.frequency);
                            ui.slider("Volume", 0.0, 1.5, &mut p.volume);
                            ui.separator();
                            ui.text("Harmonics (Character)");
                            ui.slider("3rd Harmonic (150Hz)", 0.0, 1.5, &mut p.harmonic_3);
                            ui.slider("5th Harmonic (250Hz)", 0.0, 1.5, &mut p.harmonic_5);
                            ui.slider("7th Harmonic (350Hz)", 0.0, 1.5, &mut p.harmonic_7);
                            ui.separator();
                            ui.text("Industrial Texture");
                            ui.slider("Mechanical Hum (100Hz)", 0.0, 1.5, &mut p.mechanical_hum);
                            ui.slider("Metallic Resonance", 0.0, 1.5, &mut p.metallic_res);
                            ui.separator();
                            ui.text("Character");
                            ui.slider("Saturation (Drive)", 0.0, 1.5, &mut p.saturation);
                            ui.slider("Reverb (Space)", 0.0, 1.5, &mut p.reverb);
                            ui.slider("Noise / Crackle", 0.0, 0.8, &mut p.noise);
                        });

                    // --- Animation ---
                    ui.window("DANGER ZONE")
                        .position([w * 0.60, 0.0], imgui::Condition::Always)
                        .size([w * 0.40, h], imgui::Condition::Always)
                        .resizable(false)
                        .movable(false)
                        .collapsible(false)
                        .title_bar(true)
                        .build(|| {
                            let draw_list = ui.get_window_draw_list();
                            let p_min = ui.cursor_screen_pos();
                            let canvas_size = ui.content_region_avail();
                            let center_x = p_min[0] + canvas_size[0] / 2.0;
                            let center_y = p_min[1] + canvas_size[1] / 2.0;

                            // Helper for color packing
                            let pack_color = |r: f32, g: f32, b: f32, a: f32| -> u32 {
                                let r = (r * 255.0) as u32;
                                let g = (g * 255.0) as u32;
                                let b = (b * 255.0) as u32;
                                let a = (a * 255.0) as u32;
                                (a << 24) | (b << 16) | (g << 8) | r
                            };

                            // Draw Transformer Box
                            let box_color = pack_color(0.3, 0.3, 0.3, 1.0);
                            draw_list
                                .add_rect(
                                    [p_min[0] + 20.0, center_y - 50.0],
                                    [p_min[0] + 120.0, center_y + 100.0],
                                    box_color,
                                )
                                .filled(true)
                                .build();

                            draw_list.add_text(
                                [p_min[0] + 30.0, center_y],
                                pack_color(1.0, 1.0, 0.0, 1.0),
                                "НЕ ЛЕЗЬ!\nУБЬЁТЬ!",
                            );

                            // Draw Person
                            let person_color = pack_color(1.0, 1.0, 1.0, 1.0);
                            let mut rng = rand::thread_rng();

                            // Jitter if playing
                            let jitter = if p.is_playing { 5.0 } else { 0.0 };
                            // Helper to get jitter
                            let get_jitter =
                                |rng: &mut rand::rngs::ThreadRng| rng.gen_range(-jitter..jitter);

                            let head_pos = [
                                center_x + 100.0 + get_jitter(&mut rng),
                                center_y - 50.0 + get_jitter(&mut rng),
                            ];

                            // Head
                            draw_list.add_circle(head_pos, 15.0, person_color).build();
                            // Body
                            draw_list
                                .add_line(
                                    [head_pos[0], head_pos[1] + 15.0],
                                    [head_pos[0], head_pos[1] + 80.0],
                                    person_color,
                                )
                                .thickness(3.0)
                                .build();
                            // Arms
                            draw_list
                                .add_line(
                                    [head_pos[0], head_pos[1] + 30.0],
                                    [
                                        head_pos[0] - 30.0 + get_jitter(&mut rng),
                                        head_pos[1] + 50.0 + get_jitter(&mut rng),
                                    ],
                                    person_color,
                                )
                                .thickness(3.0)
                                .build();
                            draw_list
                                .add_line(
                                    [head_pos[0], head_pos[1] + 30.0],
                                    [
                                        head_pos[0] + 30.0 + get_jitter(&mut rng),
                                        head_pos[1] + 50.0 + get_jitter(&mut rng),
                                    ],
                                    person_color,
                                )
                                .thickness(3.0)
                                .build();
                            // Legs
                            draw_list
                                .add_line(
                                    [head_pos[0], head_pos[1] + 80.0],
                                    [
                                        head_pos[0] - 20.0 + get_jitter(&mut rng),
                                        head_pos[1] + 130.0 + get_jitter(&mut rng),
                                    ],
                                    person_color,
                                )
                                .thickness(3.0)
                                .build();
                            draw_list
                                .add_line(
                                    [head_pos[0], head_pos[1] + 80.0],
                                    [
                                        head_pos[0] + 20.0 + get_jitter(&mut rng),
                                        head_pos[1] + 130.0 + get_jitter(&mut rng),
                                    ],
                                    person_color,
                                )
                                .thickness(3.0)
                                .build();

                            // Lightning
                            if p.is_playing {
                                let lightning_color = pack_color(0.5, 0.8, 1.0, 1.0);
                                let num_bolts = rng.gen_range(1..4);

                                for _ in 0..num_bolts {
                                    let start = [
                                        p_min[0] + 120.0,
                                        center_y - 20.0 + rng.gen_range(-20.0..20.0),
                                    ];
                                    let end = head_pos;

                                    let mut points = vec![start];
                                    let segments = 5;
                                    for i in 1..segments {
                                        let t = i as f32 / segments as f32;
                                        let mx = start[0] + (end[0] - start[0]) * t;
                                        let my = start[1] + (end[1] - start[1]) * t;
                                        points.push([
                                            mx + rng.gen_range(-20.0..20.0),
                                            my + rng.gen_range(-20.0..20.0),
                                        ]);
                                    }
                                    points.push(end);

                                    draw_list
                                        .add_polyline(points, lightning_color)
                                        .thickness(2.0)
                                        .build();
                                }
                            }
                        });
                }

                let gl_window = display.gl_window();
                let mut target = display.draw();
                target.clear_color(0.1, 0.1, 0.1, 1.0);

                platform.prepare_render(ui, gl_window.window());
                let draw_data = imgui.render();
                renderer
                    .render(&mut target, draw_data)
                    .expect("Rendering failed");
                target.finish().expect("Failed to swap buffers");
            }
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }
            _ => (),
        }
    });
}

fn setup_audio(params: Arc<Mutex<AudioParams>>) -> anyhow::Result<cpal::Stream> {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .context("No output device available")?;

    // Try to get the best config
    let config = device.default_output_config()?;
    println!("Audio config: {:?}", config);

    let sample_rate = config.sample_rate().0 as f32;
    let channels = config.channels() as usize;

    let mut phase = 0.0;
    let mut lfo_phase = 0.0;
    // Create reverb state (Stereo Delay)
    let mut reverb = ReverbState::new(sample_rate);

    let err_fn = |err| eprintln!("an error occurred on stream: {}", err);

    let stream = match config.sample_format() {
        cpal::SampleFormat::F32 => device.build_output_stream(
            &config.into(),
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                // We need a local RNG for the audio thread
                let mut local_rng = rand::thread_rng();
                write_audio(
                    data,
                    channels,
                    &mut phase,
                    &mut lfo_phase,
                    &mut reverb,
                    sample_rate,
                    &params,
                    &mut local_rng,
                )
            },
            err_fn,
            None,
        )?,
        _ => {
            return Err(anyhow::anyhow!(
                "Unsupported sample format - this example only supports F32"
            ))
        }
    };

    stream.play()?;
    Ok(stream)
}

fn write_audio(
    output: &mut [f32],
    channels: usize,
    phase: &mut f32,
    lfo_phase: &mut f32,
    reverb: &mut ReverbState,
    sample_rate: f32,
    params: &Arc<Mutex<AudioParams>>,
    rng: &mut rand::rngs::ThreadRng,
) {
    // Try to lock, if we can't, just output silence or previous frame (here we just skip)
    // In real audio apps, we'd use atomics or a lock-free queue.
    // But for this simple case, a mutex is usually fast enough.
    let p = match params.lock() {
        Ok(guard) => guard,
        Err(_) => return, // Poisoned lock
    };

    if !p.is_playing {
        for sample in output.iter_mut() {
            *sample = 0.0;
        }
        return;
    }

    let freq = p.frequency;
    let vol = p.volume;
    let h3 = p.harmonic_3;
    let h5 = p.harmonic_5;
    let h7 = p.harmonic_7;
    let noise_level = p.noise;
    let saturation = p.saturation;
    let mech_hum = p.mechanical_hum;
    let metal_res = p.metallic_res;
    let reverb_mix = p.reverb;

    let phase_inc = freq * 2.0 * std::f32::consts::PI / sample_rate;

    // Metallic resonance frequency (fixed, non-harmonic)
    // 50Hz * 8.4 = 420Hz (approx) - creates a hollow metallic ring
    let metal_freq = 420.0;
    let metal_inc = metal_freq * 2.0 * std::f32::consts::PI / sample_rate;

    for frame in output.chunks_mut(channels) {
        // Additive synthesis
        let mut sample = 0.0;

        // 1. Fundamental (50Hz) - The "Body"
        // Slightly saturated sine
        let fund = (*phase).sin();
        sample += fund;

        // 2. Mechanical Hum (100Hz) - The "Rattle"
        // This is often stronger than the fundamental in acoustic noise
        // Use a Triangle wave for a "buzzier" but still deep sound
        let phase_2 = (*phase * 2.0) % (2.0 * std::f32::consts::PI);
        let triangle_2 = (phase_2 / std::f32::consts::PI) - 1.0; // Sawtooth-ish
                                                                 // Make it triangle: 2 * abs(2 * (t - floor(t + 0.5))) - 1
                                                                 // Let's stick to a soft sawtooth/triangle hybrid for "grit"
        sample += triangle_2 * mech_hum;

        // 3. Harmonics (Odd harmonics for magnetic saturation)
        sample += (*phase * 3.0).sin() * h3;
        sample += (*phase * 5.0).sin() * h5;
        sample += (*phase * 7.0).sin() * h7;

        // 4. Metallic Resonance (Ring Modulated Noise)
        // This simulates the box vibrating
        if metal_res > 0.0 {
            // Ring mod: Noise * Sine
            let metal_tone = (*lfo_phase).sin(); // We reuse lfo_phase for metal freq
            let metal_noise = rng.gen_range(-1.0..1.0) * metal_tone;
            sample += metal_noise * metal_res;
        }

        // 5. Crackle / Grit
        if noise_level > 0.0 {
            // Occasional larger impulses for "crackle"
            if rng.gen_bool(0.001) {
                sample += rng.gen_range(-1.0..1.0) * noise_level * 5.0;
            }
            // Constant background hiss
            sample += rng.gen_range(-0.2..0.2) * noise_level;
        }

        // Apply Saturation (Drive into tanh)
        // Base gain 1.0, max gain 5.0
        let drive = 1.0 + saturation * 5.0;
        sample *= drive;

        // Soft clip to prevent harsh distortion and simulate magnetic saturation
        sample = sample.tanh();

        // Apply Reverb (Stereo Delay)
        let (out_l, out_r) = reverb.process(sample, reverb_mix);

        // Normalize roughly based on drive to keep volume somewhat consistent
        let final_l = out_l * 0.4 * vol;
        let final_r = out_r * 0.4 * vol;

        if channels >= 2 {
            frame[0] = final_l;
            frame[1] = final_r;
        } else {
            for channel in frame.iter_mut() {
                *channel = (final_l + final_r) * 0.5;
            }
        }

        *phase += phase_inc;
        if *phase > 2.0 * std::f32::consts::PI {
            *phase -= 2.0 * std::f32::consts::PI;
        }

        // Update metal phase
        *lfo_phase += metal_inc;
        if *lfo_phase > 2.0 * std::f32::consts::PI {
            *lfo_phase -= 2.0 * std::f32::consts::PI;
        }
    }
}
