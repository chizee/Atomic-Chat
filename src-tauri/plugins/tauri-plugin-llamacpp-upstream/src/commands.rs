use base64::{engine::general_purpose, Engine as _};
use hmac::{Hmac, Mac};
use sha2::Sha256;
use std::collections::HashMap;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;
use tauri::{Emitter, Manager, Runtime, State};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::sync::{mpsc, Mutex};
use tokio::time::Instant;

use crate::args::{ArgumentBuilder, LlamacppConfig};
use crate::device::{get_devices_from_backend, DeviceInfo};
use crate::error::{ErrorCode, LlamacppError, ServerError, ServerResult};
use crate::path::{validate_binary_path, validate_mmproj_path, validate_model_path};
use crate::process::{
    find_session_by_model_id, get_all_active_sessions, get_all_loaded_model_ids,
    get_random_available_port, is_process_running_by_pid,
};
use crate::state::{LLamaBackendSession, LlamacppState, SessionInfo};
use jan_utils::{
    add_cuda_paths, binary_requires_cuda, setup_library_path, setup_windows_process_flags,
};

#[cfg(unix)]
use crate::process::graceful_terminate_process;

#[cfg(all(windows, target_arch = "x86_64"))]
use crate::process::force_terminate_process;

type HmacSha256 = Hmac<Sha256>;

#[derive(serde::Serialize, serde::Deserialize)]
pub struct UnloadResult {
    success: bool,
    error: Option<String>,
}

/// Core model loading logic usable without an AppHandle (CLI / test support).
pub async fn load_llama_model_impl(
    process_map_arc: Arc<Mutex<HashMap<i32, LLamaBackendSession>>>,
    backend_path: &str,
    model_id: String,
    model_path: String,
    port: u16,
    config: LlamacppConfig,
    envs: HashMap<String, String>,
    mmproj_path: Option<String>,
    is_embedding: bool,
    timeout: u64,
) -> ServerResult<SessionInfo> {
    log::info!("Attempting to launch server at path: {:?}", backend_path);
    log::info!("Using configuration: {:?}", config);

    let bin_path = validate_binary_path(backend_path)?;

    // Build arguments using the ArgumentBuilder
    let builder = ArgumentBuilder::new(config.clone(), is_embedding)
        .map_err(|e| ServerError::InvalidArgument(e))?;

    let mut args = builder.build(&model_id, &model_path, port, mmproj_path.clone());

    log::info!("Generated arguments: {:?}", args);

    // Validate paths
    let model_path_pb = validate_model_path(&mut args)?;
    let mmproj_path_pb = validate_mmproj_path(&mut args)?;

    let mmproj_path_string = if let Some(ref _mmproj_pb) = mmproj_path_pb {
        // Find the actual mmproj path from args after validation/conversion
        if let Some(mmproj_index) = args.iter().position(|arg| arg == "--mmproj") {
            Some(args[mmproj_index + 1].clone())
        } else {
            None
        }
    } else {
        None
    };

    log::info!(
        "MMPROJ Path string: {}",
        &mmproj_path_string.as_ref().unwrap_or(&"None".to_string())
    );

    let api_key: String = envs
        .get("LLAMA_API_KEY")
        .map(|s| s.to_string())
        .unwrap_or_default();

    // Configure the command to run the server
    let mut command = Command::new(&bin_path);

    command.args(args);
    command.envs(envs);

    command.stdout(Stdio::piped());
    command.stderr(Stdio::piped());
    setup_windows_process_flags(&mut command);

    // Try to add CUDA paths (works on both Windows and Linux)
    let cuda_found = add_cuda_paths(&mut command);

    // Optionally check if binary needs CUDA
    if !cuda_found && binary_requires_cuda(&bin_path) {
        log::warn!(
            "llama.cpp backend appears to require CUDA, but CUDA not found. Process may fail to start. Please install cuda runtime and try again!"
        );
    }

    // Add the binary's directory to library path
    setup_library_path(bin_path.parent(), &mut command);

    // Spawn the child process
    let mut child = command.spawn().map_err(ServerError::Io)?;

    let stderr = child.stderr.take().expect("stderr was piped");
    let stdout = child.stdout.take().expect("stdout was piped");

    // Create channels for communication between tasks
    let (ready_tx, mut ready_rx) = mpsc::channel::<bool>(1);

    // Spawn task to monitor stdout for readiness
    let stdout_ready_tx = ready_tx.clone();
    let _stdout_task = tokio::spawn(async move {
        let mut reader = BufReader::new(stdout);
        let mut byte_buffer = Vec::new();

        loop {
            byte_buffer.clear();
            match reader.read_until(b'\n', &mut byte_buffer).await {
                Ok(0) => break, // EOF
                Ok(_) => {
                    let line = String::from_utf8_lossy(&byte_buffer);
                    let line = line.trim_end();
                    if !line.is_empty() {
                        log::info!("[llamacpp stdout] {}", line);
                    }

                    // Check for readiness indicators
                    let line_lower = line.to_lowercase();
                    if line_lower.contains("http server listening")
                        || line_lower.contains("all slots are idle")
                        || line_lower.contains("starting the main loop")
                    {
                        log::info!("Server appears to be ready based on stdout: '{}'", line);
                        let _ = stdout_ready_tx.send(true).await;
                    }
                }
                Err(e) => {
                    log::error!("Error reading stdout: {}", e);
                    break;
                }
            }
        }
    });

    // Spawn task to capture stderr and monitor for errors
    let stderr_task = tokio::spawn(async move {
        let mut reader = BufReader::new(stderr);
        let mut byte_buffer = Vec::new();
        let mut stderr_buffer = String::new();

        loop {
            byte_buffer.clear();
            match reader.read_until(b'\n', &mut byte_buffer).await {
                Ok(0) => break, // EOF
                Ok(_) => {
                    let line = String::from_utf8_lossy(&byte_buffer);
                    let line = line.trim_end();

                    if !line.is_empty() {
                        stderr_buffer.push_str(line);
                        stderr_buffer.push('\n');
                        log::info!("[llamacpp] {}", line);

                        // Check for readiness indicator
                        let line_lower = line.to_string().to_lowercase();
                        if line_lower.contains("server is listening on")
                            || line_lower.contains("starting the main loop")
                            || line_lower.contains("server listening on")
                        {
                            log::info!("Model appears to be ready based on logs: '{}'", line);
                            let _ = ready_tx.send(true).await;
                        }
                    }
                }
                Err(e) => {
                    log::error!("Error reading logs: {}", e);
                    break;
                }
            }
        }

        stderr_buffer
    });

    // Check if process exited early
    if let Some(status) = child.try_wait()? {
        if !status.success() {
            let stderr_output = stderr_task.await.unwrap_or_default();
            // WS1.1/WS3.2: warn! (not error!) so the SentryLogger bridge does not
            // raise a duplicate crash event — the structured error returned below
            // is reported once by the frontend model-load choke point — and
            // classify native crash exit codes into an actionable error.
            log::warn!("llama.cpp failed early with code {:?}", status);
            log::warn!("{}", stderr_output);
            return Err(LlamacppError::from_exit_status(&status, &stderr_output).into());
        }
    }

    // Wait for server to be ready or timeout
    let timeout_duration = Duration::from_secs(timeout);
    let start_time = Instant::now();
    log::info!("Waiting for model session to be ready...");

    loop {
        tokio::select! {
            // Server is ready
            Some(true) = ready_rx.recv() => {
                log::info!("Model is ready to accept requests!");
                break;
            }
            // Check for process exit more frequently
            _ = tokio::time::sleep(Duration::from_millis(50)) => {
                // Check if process exited
                if let Some(status) = child.try_wait()? {
                    let stderr_output = stderr_task.await.unwrap_or_default();
                    if !status.success() {
                        // WS1.1: warn! (not error!) — the structured error returned
                        // below is reported once by the frontend choke point, so an
                        // error! here is a duplicate Sentry crash event.
                        // WS3.2: classify native crash exit codes (access violation /
                        // segfault) into an actionable, recoverable error.
                        log::warn!("llama.cpp exited with error code {:?}", status);
                        return Err(LlamacppError::from_exit_status(&status, &stderr_output).into());
                    } else {
                        log::warn!("llama.cpp exited successfully but without ready signal");
                        return Err(LlamacppError::from_stderr(&stderr_output).into());
                    }
                }

                // Timeout check
                if start_time.elapsed() > timeout_duration {
                    log::error!("Timeout waiting for server to be ready");
                    let _ = child.kill().await;
                    let stderr_output = stderr_task.await.unwrap_or_default();
                    return Err(LlamacppError::new(
                        ErrorCode::ModelLoadTimedOut,
                        "The model took too long to load and timed out.".into(),
                        Some(format!("Timeout: {}s\n\nStderr:\n{}", timeout_duration.as_secs(), stderr_output)),
                    ).into());
                }
            }
        }
    }

    // Get the PID to use as session ID
    let pid = child.id().map(|id| id as i32).unwrap_or(-1);

    log::info!("Server process started with PID: {} and is ready", pid);
    let session_info = SessionInfo {
        pid: pid.clone(),
        port: port.into(),
        model_id: model_id,
        model_path: model_path_pb.display().to_string(),
        is_embedding: is_embedding,
        api_key: api_key,
        mmproj_path: mmproj_path_string,
    };

    {
        let mut process_map = process_map_arc.lock().await;
        process_map.insert(
            pid.clone(),
            LLamaBackendSession {
                child,
                info: session_info.clone(),
            },
        );
    }

    Ok(session_info)
}

/// Tauri event emitted when a llama-server child process that was running
/// (i.e. had already loaded a model) exits unexpectedly during generation.
/// Payload: `{ model_id, pid, error_code, message }`.
pub const SESSION_DIED_EVENT: &str = "local_backend://llamacpp_upstream_session_died";

/// Load a llama model and start the server
#[tauri::command]
pub async fn load_llama_model<R: Runtime>(
    app_handle: tauri::AppHandle<R>,
    backend_path: &str,
    model_id: String,
    model_path: String,
    port: u16,
    config: LlamacppConfig,
    envs: HashMap<String, String>,
    mmproj_path: Option<String>,
    is_embedding: bool,
    timeout: u64,
) -> ServerResult<SessionInfo> {
    let state: State<LlamacppState> = app_handle.state();
    let session_info = load_llama_model_impl(
        state.llama_server_process.clone(),
        backend_path,
        model_id,
        model_path,
        port,
        config,
        envs,
        mmproj_path,
        is_embedding,
        timeout,
    )
    .await?;

    // Spawn a background watcher task that detects unexpected process exits
    // (crashes during generation). Without this watcher, a Vulkan or other
    // backend crash that happens AFTER the model loads is invisible: no exit
    // code is classified, no Sentry event fires, and the user only sees a
    // broken HTTP stream with no actionable message.
    //
    // The watcher polls `try_wait()` (non-blocking) on the child every 500 ms.
    // When the process exits unexpectedly it:
    //   1. Removes the session from the process_map (so unload is a no-op).
    //   2. Classifies the exit via `from_exit_status` (SIGSEGV/SIGABRT etc.).
    //   3. Logs at `error!` so the Sentry logger bridge forwards it as a crash event.
    //   4. Emits `SESSION_DIED_EVENT` so the extension and web-app can surface
    //      an actionable "model crashed during generation" message.
    //
    // Intentional unloads are transparent: `unload_llama_model` removes the
    // entry from the map, so the watcher finds `None` on its next poll and exits.
    let pid = session_info.pid;
    let process_map_watcher = state.llama_server_process.clone();
    let app_handle_watcher = app_handle.clone();
    tokio::spawn(async move {
        const POLL_INTERVAL: Duration = Duration::from_millis(500);
        loop {
            tokio::time::sleep(POLL_INTERVAL).await;

            // Scope the lock acquisition tightly so we do not hold it across
            // the sleep. try_wait() is synchronous and non-blocking.
            let poll_outcome: Result<Option<(std::process::ExitStatus, SessionInfo)>, ()> = {
                let mut map = process_map_watcher.lock().await;
                match map.get_mut(&pid) {
                    None => Err(()), // Session intentionally removed by unload
                    Some(session) => {
                        match session.child.try_wait() {
                            Ok(None) => Ok(None), // Still running
                            Ok(Some(status)) => {
                                let info = session.info.clone();
                                // Remove from map so subsequent unload calls are no-ops
                                map.remove(&pid);
                                Ok(Some((status, info)))
                            }
                            Err(e) => {
                                log::warn!(
                                    "llamacpp-upstream watcher: try_wait error for PID {}: {}",
                                    pid,
                                    e
                                );
                                Err(())
                            }
                        }
                    }
                }
            }; // lock released here

            match poll_outcome {
                Ok(None) => {} // Still running — continue polling
                Err(()) => break,
                Ok(Some((status, info))) => {
                    // Unexpected exit: classify and report.
                    let error = LlamacppError::from_exit_status(&status, "");
                    // log::error! → Sentry logger bridge (ATO-244: generation-time
                    // crashes were previously invisible to Sentry because the
                    // classification path was only wired to the load path).
                    log::error!(
                        "llamacpp-upstream: llama-server (PID {}, model='{}') exited \
                         unexpectedly during generation — code={:?} — {}",
                        pid,
                        info.model_id,
                        status.code(),
                        error.message
                    );

                    #[derive(serde::Serialize)]
                    struct SessionDiedPayload {
                        model_id: String,
                        pid: i32,
                        error_code: String,
                        message: String,
                    }
                    let payload = SessionDiedPayload {
                        model_id: info.model_id.clone(),
                        pid: info.pid,
                        error_code: format!("{:?}", error.code),
                        message: error.message.clone(),
                    };
                    if let Err(e) =
                        app_handle_watcher.emit(SESSION_DIED_EVENT, &payload)
                    {
                        log::warn!(
                            "llamacpp-upstream watcher: failed to emit {} event: {}",
                            SESSION_DIED_EVENT,
                            e
                        );
                    }
                    break;
                }
            }
        }
    });

    Ok(session_info)
}

/// Unload a llama model by terminating its process
#[tauri::command]
pub async fn unload_llama_model<R: Runtime>(
    app_handle: tauri::AppHandle<R>,
    pid: i32,
) -> ServerResult<UnloadResult> {
    let state: State<LlamacppState> = app_handle.state();
    let mut map = state.llama_server_process.lock().await;

    if let Some(session) = map.remove(&pid) {
        let mut child = session.child;

        #[cfg(unix)]
        {
            graceful_terminate_process(&mut child).await;
        }

        #[cfg(all(windows, target_arch = "x86_64"))]
        {
            force_terminate_process(&mut child).await;
        }

        Ok(UnloadResult {
            success: true,
            error: None,
        })
    } else {
        log::warn!("No server with PID '{}' found", pid);
        Ok(UnloadResult {
            success: true,
            error: None,
        })
    }
}

/// Get available devices from the llama.cpp backend
#[tauri::command]
pub async fn get_devices(
    backend_path: &str,
    envs: HashMap<String, String>,
) -> ServerResult<Vec<DeviceInfo>> {
    get_devices_from_backend(backend_path, envs).await
}

/// Generate API key using HMAC-SHA256
#[tauri::command]
pub fn generate_api_key(model_id: String, api_secret: String) -> Result<String, String> {
    let mut mac = HmacSha256::new_from_slice(api_secret.as_bytes())
        .map_err(|e| format!("Invalid key length: {}", e))?;
    mac.update(model_id.as_bytes());
    let result = mac.finalize();
    let code_bytes = result.into_bytes();
    let hash = general_purpose::STANDARD.encode(code_bytes);
    Ok(hash)
}

/// Check if a process is still running
#[tauri::command]
pub async fn is_process_running<R: Runtime>(
    app_handle: tauri::AppHandle<R>,
    pid: i32,
) -> Result<bool, String> {
    is_process_running_by_pid(app_handle, pid).await
}

/// Get a random available port
#[tauri::command]
pub async fn get_random_port<R: Runtime>(app_handle: tauri::AppHandle<R>) -> Result<u16, String> {
    get_random_available_port(app_handle).await
}

/// Find session information by model ID
#[tauri::command]
pub async fn find_session_by_model<R: Runtime>(
    app_handle: tauri::AppHandle<R>,
    model_id: String,
) -> Result<Option<SessionInfo>, String> {
    find_session_by_model_id(app_handle, &model_id).await
}

/// Get all loaded model IDs
#[tauri::command]
pub async fn get_loaded_models<R: Runtime>(
    app_handle: tauri::AppHandle<R>,
) -> Result<Vec<String>, String> {
    get_all_loaded_model_ids(app_handle).await
}

/// Get all active sessions
#[tauri::command]
pub async fn get_all_sessions<R: Runtime>(
    app_handle: tauri::AppHandle<R>,
) -> Result<Vec<SessionInfo>, String> {
    get_all_active_sessions(app_handle).await
}

/// Get session information by model ID
#[tauri::command]
pub async fn get_session_by_model<R: Runtime>(
    app_handle: tauri::AppHandle<R>,
    model_id: String,
) -> Result<Option<SessionInfo>, String> {
    find_session_by_model_id(app_handle, &model_id).await
}
