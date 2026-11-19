// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::sync::{Arc, Mutex};
use tauri::{Manager, RunEvent, WindowEvent};
use tauri_plugin_shell::process::CommandChild;
use tauri_plugin_shell::ShellExt;

#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;

#[cfg(target_os = "windows")]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

/// Managed Tauri state holding the active `comic-backend.exe` sidecar process handle.
struct SidecarProcessState(Arc<Mutex<Option<CommandChild>>>);

/// Terminate the `comic-backend.exe` sidecar process and its entire child process tree
/// (both the PyInstaller bootloader and unpacked Uvicorn worker) cleanly on window exit.
fn terminate_sidecar(state: &SidecarProcessState) {
    if let Ok(mut guard) = state.0.lock() {
        if let Some(child) = guard.take() {
            let pid = child.pid();

            #[cfg(target_os = "windows")]
            {
                let _ = std::process::Command::new("taskkill")
                    .args(["/F", "/T", "/PID", &pid.to_string()])
                    .creation_flags(CREATE_NO_WINDOW)
                    .status();
            }

            let _ = child.kill();
        }
    }
}

fn main() {
    let sidecar_state = SidecarProcessState(Arc::new(Mutex::new(None)));

    let app = tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(sidecar_state)
        .setup(|app| {
            // Spawn `comic-backend.exe` silently as a Tauri Sidecar Process on startup
            let sidecar_command = app
                .shell()
                .sidecar("comic-backend")
                .map_err(|e| format!("Failed to configure comic-backend sidecar: {e}"))?;

            match sidecar_command.spawn() {
                Ok((_rx, child)) => {
                    let state = app.state::<SidecarProcessState>();
                    if let Ok(mut guard) = state.0.lock() {
                        *guard = Some(child);
                    };
                }
                Err(err) => {
                    eprintln!("[Tauri Sidecar] Warning: could not spawn comic-backend.exe: {err}");
                }
            }

            Ok(())
        })
        .on_window_event(|window, event| {
            if matches!(event, WindowEvent::CloseRequested { .. } | WindowEvent::Destroyed) {
                let state = window.state::<SidecarProcessState>();
                terminate_sidecar(&state);
            }
        })
        .build(tauri::generate_context!())
        .expect("error while building Tauri application");

    app.run(|app_handle, event| {
        if matches!(event, RunEvent::ExitRequested { .. } | RunEvent::Exit) {
            let state = app_handle.state::<SidecarProcessState>();
            terminate_sidecar(&state);
        }
    });
}
