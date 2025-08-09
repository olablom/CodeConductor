// Prevents additional console output on launch of Windows Subsystem for Linux (WSL) distros
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::{
    fs,
    io::{self, Read},
    path::{Path, PathBuf},
    process::Command,
    sync::Mutex,
};
use tauri::State;
use serde_json::{self, Value as JsonValue};

// App state
struct AppState {
    generation_count: Mutex<u32>,
}

#[tauri::command]
async fn test_tauri() -> Result<String, String> {
    Ok("Tauri IPC is working!".to_string())
}

#[tauri::command]
async fn generate_code(prompt: String, state: State<'_, AppState>) -> Result<String, String> {
    // Increment generation count
    {
        let mut count = state.generation_count.lock().unwrap();
        *count += 1;
    }
    
    // Generate code based on prompt content
    let generated_code = match prompt.to_lowercase() {
        p if p.contains("python") && p.contains("sort") => {
            r#"def sort_list(numbers):
    """Sort a list of numbers in ascending order."""
    return sorted(numbers)

# Example usage
numbers = [3, 1, 4, 1, 5, 9, 2, 6]
sorted_numbers = sort_list(numbers)
print(f"Original: {numbers}")
print(f"Sorted: {sorted_numbers}")"#.to_string()
        },
        p if p.contains("python") && p.contains("function") => {
            r#"def example_function():
    """A sample Python function."""
    result = "Hello from CodeConductor!"
    print(result)
    return result

# Call the function
output = example_function()
print(f"Function returned: {output}")"#.to_string()
        },
        p if p.contains("javascript") || p.contains("js") => {
            r#"function exampleFunction() {
    const message = "Hello from CodeConductor!";
    console.log(message);
    return message;
}

// Call the function
const result = exampleFunction();
console.log(`Function returned: ${result}`);"#.to_string()
        },
        p if p.contains("rust") => {
            r#"fn example_function() -> String {
    let message = "Hello from CodeConductor!";
    println!("{}", message);
    message.to_string()
}

fn main() {
    let result = example_function();
    println!("Function returned: {}", result);
}"#.to_string()
        },
        _ => {
            format!(r#"// Generated code for: "{}"
// This is a placeholder implementation

function example() {{
    console.log("Hello from CodeConductor!")
    return "Generated successfully"
}}

// Usage
const result = example()
console.log("Result:", result)"#, prompt)
        }
    };
    
    Ok(generated_code)
}

#[tauri::command]
async fn get_generation_stats(state: State<'_, AppState>) -> Result<u32, String> {
    let count = state.generation_count.lock().unwrap();
    Ok(*count)
}

#[tauri::command]
async fn save_code_to_file(_code: String) -> Result<String, String> {
    // TODO: Implement file saving
    Ok("Code saved successfully".to_string())
}

#[tauri::command]
async fn open_external_url(url: String) -> Result<(), String> {
    // TODO: Implement external URL opening
    println!("Would open URL: {}", url);
    Ok(())
}

fn artifacts_root() -> PathBuf {
    PathBuf::from("artifacts")
}

fn latest_run_dir() -> io::Result<PathBuf> {
    let runs_dir = artifacts_root().join("runs");
    if !runs_dir.exists() {
        return Err(io::Error::new(io::ErrorKind::NotFound, "runs dir not found"));
    }
    let mut entries: Vec<PathBuf> = fs::read_dir(runs_dir)?
        .filter_map(|e| e.ok().map(|d| d.path()))
        .filter(|p| p.is_dir())
        .collect();
    entries.sort_by(|a, b| b.file_name().cmp(&a.file_name()));
    entries
        .into_iter()
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "no runs found"))
}

fn read_json_file(path: &Path) -> Result<JsonValue, String> {
    let mut f = fs::File::open(path).map_err(|e| format!("open {:?}: {}", path, e))?;
    let mut s = String::new();
    f.read_to_string(&mut s)
        .map_err(|e| format!("read {:?}: {}", path, e))?;
    serde_json::from_str::<JsonValue>(&s).map_err(|e| format!("parse {:?}: {}", path, e))
}

#[tauri::command]
async fn get_latest_selector_decision() -> Result<JsonValue, String> {
    let dir = latest_run_dir().map_err(|e| e.to_string())?;
    let path = dir.join("selector_decision.json");
    if !path.exists() {
        return Err("selector_decision.json not found".into());
    }
    read_json_file(&path)
}

#[tauri::command]
async fn get_latest_consensus() -> Result<JsonValue, String> {
    let dir = latest_run_dir().map_err(|e| e.to_string())?;
    let path = dir.join("consensus.json");
    if !path.exists() {
        return Err("consensus.json not found".into());
    }
    read_json_file(&path)
}

#[tauri::command]
async fn run_warmup_smoke() -> Result<JsonValue, String> {
    // Spawn python scripts/warmup_smoke.py --start-server
    let output = Command::new("python")
        .arg("scripts/warmup_smoke.py")
        .arg("--start-server")
        .output()
        .map_err(|e| format!("failed to run warmup_smoke: {}", e))?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).to_string());
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Try to parse last JSON line from stdout
    let line = stdout
        .lines()
        .last()
        .ok_or_else(|| "no output from warmup_smoke".to_string())?;
    let v: JsonValue = serde_json::from_str(line).map_err(|e| e.to_string())?;
    Ok(v)
}

#[tauri::command]
async fn get_warmup_history(limit: usize) -> Result<JsonValue, String> {
    let dir = artifacts_root().join("latency");
    if !dir.exists() {
        return Ok(JsonValue::Array(vec![]));
    }
    let mut files: Vec<PathBuf> = fs::read_dir(&dir)
        .map_err(|e| e.to_string())?
        .filter_map(|e| e.ok().map(|d| d.path()))
        .filter(|p| p.file_name().map(|n| n.to_string_lossy().starts_with("warmup_")).unwrap_or(false))
        .collect();
    files.sort_by(|a, b| b.file_name().cmp(&a.file_name()));
    let mut out = Vec::new();
    for p in files.into_iter().take(limit.max(1)) {
        if let Ok(val) = read_json_file(&p) {
            out.push(val);
        }
    }
    Ok(JsonValue::Array(out))
}

fn main() {
    let app_state = AppState {
        generation_count: Mutex::new(0),
    };

    tauri::Builder::default()
        .manage(app_state)
        .invoke_handler(tauri::generate_handler![
            test_tauri,
            generate_code,
            get_generation_stats,
            save_code_to_file,
            open_external_url,
            get_latest_selector_decision,
            get_latest_consensus,
            run_warmup_smoke,
            get_warmup_history
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
} 