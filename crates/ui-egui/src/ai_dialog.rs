//! Model context for Edit › Generate Image and Edit › Generative Fill.
//!
//! The model itself is the command's menu. This adds the pictures sent with the prompt: an open
//! document or one of its layers, a file from the import dialog, a file dropped on the window, or
//! an http(s) link. At most [`photocraft_engine::ai_cmds::MAX_CONTEXT`] pictures.

use serde_json::{Map, Value, json};

use crate::PhotocraftApp;
use crate::theme::Tokens;

const IMAGE_EXT: &[&str] = &["png", "jpg", "jpeg", "webp", "gif", "tif", "tiff", "bmp"];

pub(crate) fn refresh_workspace(app: &PhotocraftApp, fields: &mut Map<String, Value>) {
    let Some(cmd) = fields.get("__command").and_then(Value::as_str) else { return };
    if !photocraft_engine::ai_cmds::is_ai_command(cmd) {
        return;
    }
    let active = app.session.active().map(|st| st.doc.id);
    let mut docs: Vec<_> = app.session.documents().iter().enumerate().collect();
    if let Some(id) = active
        && let Some(pos) = docs.iter().position(|(_, st)| st.doc.id == id)
    {
        let current = docs.remove(pos);
        docs.insert(0, current);
    }
    let mut items = Vec::new();
    for (i, st) in docs {
        if items.len() >= 64 {
            break;
        }
        items.push(json!({"id": format!("doc:{i}"), "name": st.doc.name}));
        for (_, _, layer) in st.doc.walk() {
            if items.len() >= 64 {
                break;
            }
            if layer.is_group() || layer.surface().is_none() {
                continue;
            }
            items.push(json!({
                "id": format!("layer:{i}:{}", layer.id.0),
                "name": format!("{} · {}", st.doc.name, layer.name),
            }));
        }
    }
    fields.insert("__workspace".into(), Value::Array(items));
}

/// The open AI dialog, when one is showing.
pub(crate) fn dialog_id(app: &PhotocraftApp) -> Option<u64> {
    app.ui.dialogs.iter().rev().find_map(|d| {
        let cmd = d.fields.get("__command").and_then(Value::as_str)?;
        photocraft_engine::ai_cmds::is_ai_command(cmd).then_some(d.id)
    })
}

pub(crate) fn context_controls(ui: &mut egui::Ui, fields: &mut Map<String, Value>) {
    let Some(cmd) = fields.get("__command").and_then(Value::as_str) else { return };
    if !photocraft_engine::ai_cmds::is_ai_command(cmd) {
        return;
    }
    let t = Tokens::get(ui.ctx());
    ui.add_space(8.0);
    ui.label(egui::RichText::new(tl!("Context images")).color(t.text));
    let mut context = context_of(fields);
    let mut removed = None;
    for (i, item) in context.iter().enumerate() {
        let name = item.get("name").and_then(Value::as_str).unwrap_or("image");
        ui.horizontal(|ui| {
            ui.add(egui::Label::new(egui::RichText::new(clip_name(name)).color(t.text)).truncate());
            if crate::widgets::secondary_button(ui, tl!("Remove"), 72.0).clicked() {
                removed = Some(i);
            }
        });
    }
    if let Some(i) = removed {
        context.remove(i);
        fields.insert("context".into(), Value::Array(context.clone()));
    }
    let workspace = workspace_of(fields);
    let mut pick = fields.get("__contextPick").and_then(Value::as_str).unwrap_or("").to_string();
    if !workspace.iter().any(|(id, _)| id == &pick) {
        pick = workspace.first().map(|(id, _)| id.clone()).unwrap_or_default();
    }
    ui.horizontal(|ui| {
        let opts: Vec<(String, &str)> = if workspace.is_empty() {
            vec![("".into(), tl!("None"))]
        } else {
            workspace.iter().map(|(id, name)| (id.clone(), name.as_str())).collect()
        };
        // Document and layer names are the user's, so they stay untranslated.
        crate::widgets::dropdown_literal(ui, "ai-context-doc", &mut pick, &opts, 240.0);
        let can_add = !pick.is_empty() && context.len() < photocraft_engine::ai_cmds::MAX_CONTEXT;
        if ui.add_enabled_ui(can_add, |ui| crate::widgets::secondary_button(ui, tl!("Add"), 64.0)).inner.clicked()
            && let Some((_, name)) = workspace.iter().find(|(id, _)| id == &pick)
            && let Some(item) = workspace_item(&pick, name)
        {
            context.push(item);
            fields.insert("context".into(), Value::Array(context.clone()));
        }
    });
    fields.insert("__contextPick".into(), json!(pick));
    ui.horizontal(|ui| {
        if crate::widgets::secondary_button(ui, tl!("Import…"), 84.0).clicked() {
            fields.insert("__importContext".into(), json!(true));
        }
    });
    let mut link = fields.get("__imageLink").and_then(Value::as_str).unwrap_or("").to_string();
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(tl!("Image link")).color(t.text_dim));
        ui.add(egui::TextEdit::singleline(&mut link).desired_width(180.0).hint_text("https://"));
        let can_load = acceptable_link(&link) && context.len() < photocraft_engine::ai_cmds::MAX_CONTEXT;
        if ui.add_enabled_ui(can_load, |ui| crate::widgets::secondary_button(ui, tl!("Load"), 64.0)).inner.clicked() {
            let url = link.trim().to_string();
            context.push(json!({"name": clip_name(&url), "url": url}));
            fields.insert("context".into(), Value::Array(context));
            link.clear();
        }
    });
    fields.insert("__imageLink".into(), json!(link));
    ui.add_space(4.0);
    // The dialog's maximum width would otherwise clip this sentence on one line.
    let hint_width = ui.available_width().clamp(280.0, 540.0);
    ui.add_sized(
        egui::vec2(hint_width, 36.0),
        egui::Label::new(
            egui::RichText::new(tl!("Add pictures from this workspace, import a file, drop one here, or paste a link. Generative Fill also sends the open picture."))
                .color(t.text_dim)
                .size(11.0),
        )
        .wrap(),
    );
}

/// Start the import dialog. The answer is added to dialog `id` on a later frame.
pub(crate) fn begin_import(app: &mut PhotocraftApp, dialog: u64) -> Result<serde_json::Value, String> {
    let initial_dir = crate::file_dialog::last_used_dir(&app.ui.recent_files);
    let extensions = Some(IMAGE_EXT.iter().map(|s| (*s).to_string()).collect());
    app.ask_file(crate::file_dialog::FileDialogRequest::Open { multiple: true, initial_dir, extensions }, move |app, answer| {
        match answer {
            crate::file_dialog::FileDialogAnswer::Paths(paths) => {
                for path in paths {
                    push_path(app, dialog, &crate::file_open::display_name(&path), &path)?;
                }
                Ok(Value::Null)
            }
            crate::file_dialog::FileDialogAnswer::Contents(name, bytes) => {
                push_bytes(app, dialog, &name, &bytes)?;
                Ok(Value::Null)
            }
            crate::file_dialog::FileDialogAnswer::ContentsMany(files) => {
                for (name, bytes) in files {
                    push_bytes(app, dialog, &name, &bytes)?;
                }
                Ok(Value::Null)
            }
            _ => Err("the file dialog gave an unexpected answer".into()),
        }
    })
}

pub(crate) fn take_import(app: &mut PhotocraftApp, id: u64) {
    let wants = app.ui.dialogs.iter().any(|d| d.id == id && d.fields.get("__importContext").and_then(Value::as_bool) == Some(true));
    if !wants {
        return;
    }
    if let Some(d) = app.ui.dialog_mut(id) {
        d.fields.insert("__importContext".into(), json!(false));
    }
    if let Err(e) = begin_import(app, id)
        && e != crate::file_dialog::CANCELLED
    {
        app.ui.status = e;
        app.ui.status_error = true;
    }
}

pub(crate) fn push_path(app: &mut PhotocraftApp, dialog: u64, name: &str, path: &str) -> Result<(), String> {
    if !extension_ok(name) && !extension_ok(path) {
        return Err(format!("{name} is not an image PhotoCraft can read"));
    }
    push_item(app, dialog, json!({"name": clip_name(name), "path": path}))
}

pub(crate) fn push_bytes(app: &mut PhotocraftApp, dialog: u64, name: &str, bytes: &[u8]) -> Result<(), String> {
    if bytes.is_empty() || bytes.len() > 32 << 20 {
        return Err(format!("{name} was empty or larger than 32 MB"));
    }
    if !looks_like_image(name, bytes) {
        return Err(format!("{name} is not an image PhotoCraft can read"));
    }
    let data = photocraft_paint::tile::b64_encode(bytes);
    push_item(app, dialog, json!({"name": clip_name(name), "data": data}))
}

fn push_item(app: &mut PhotocraftApp, dialog: u64, item: Value) -> Result<(), String> {
    let Some(fields) = app.ui.dialog_mut(dialog).map(|d| &mut d.fields) else {
        return Err("the image dialog is no longer open".into());
    };
    let mut context = context_of(fields);
    if context.len() >= photocraft_engine::ai_cmds::MAX_CONTEXT {
        return Err(format!("at most {} context images", photocraft_engine::ai_cmds::MAX_CONTEXT));
    }
    context.push(item);
    fields.insert("context".into(), Value::Array(context));
    Ok(())
}

fn context_of(fields: &Map<String, Value>) -> Vec<Value> {
    fields.get("context").and_then(Value::as_array).cloned().unwrap_or_default()
}

fn workspace_of(fields: &Map<String, Value>) -> Vec<(String, String)> {
    fields
        .get("__workspace")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(|item| {
                    let id = item.get("id").and_then(Value::as_str)?.to_string();
                    let name = item.get("name").and_then(Value::as_str)?.to_string();
                    Some((id, name))
                })
                .collect()
        })
        .unwrap_or_default()
}

fn workspace_item(id: &str, name: &str) -> Option<Value> {
    if let Some(rest) = id.strip_prefix("doc:") {
        let document: u64 = rest.parse().ok()?;
        return Some(json!({"name": name, "document": document}));
    }
    let rest = id.strip_prefix("layer:")?;
    let (document, layer) = rest.split_once(':')?;
    Some(json!({"name": name, "document": document.parse::<u64>().ok()?, "layer": layer.parse::<u64>().ok()?}))
}

fn acceptable_link(raw: &str) -> bool {
    let url = raw.trim();
    let Some(rest) = url.strip_prefix("https://").or_else(|| url.strip_prefix("http://")) else { return false };
    if url.len() > 2000 || rest.is_empty() || url.chars().any(|c| c.is_control() || c.is_whitespace()) {
        return false;
    }
    let host = rest.split(['/', '?', '#']).next().unwrap_or("");
    !host.is_empty() && !host.contains('@')
}

fn extension_ok(name: &str) -> bool {
    name.rsplit('.').next().is_some_and(|ext| IMAGE_EXT.iter().any(|known| ext.eq_ignore_ascii_case(known)))
}

fn looks_like_image(name: &str, bytes: &[u8]) -> bool {
    extension_ok(name)
        || bytes.starts_with(b"\x89PNG")
        || bytes.starts_with(b"\xFF\xD8")
        || bytes.starts_with(b"GIF8")
        || bytes.starts_with(b"RIFF")
        || bytes.starts_with(b"II*\x00")
        || bytes.starts_with(b"MM\x00*")
        || bytes.starts_with(b"BM")
}

fn clip_name(name: &str) -> String {
    let mut out = String::new();
    for (i, ch) in name.chars().enumerate() {
        if i >= 48 {
            out.push('…');
            break;
        }
        out.push(ch);
    }
    if out.is_empty() { "image".into() } else { out }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_ids_become_context_objects() {
        let doc = workspace_item("doc:2", "Bowl").unwrap();
        assert_eq!(doc["document"], 2);
        assert!(doc.get("layer").is_none());
        let layer = workspace_item("layer:1:44", "Zigzag · Ink").unwrap();
        assert_eq!(layer["document"], 1);
        assert_eq!(layer["layer"], 44);
        assert!(workspace_item("nope", "x").is_none());
    }

    #[test]
    fn the_generate_dialog_offers_every_image_model() {
        use crate::filter_dialog::{Kind, parse_spec};
        let spec = photocraft_engine::commands::find("edit.generateImage").unwrap().params;
        let params = parse_spec(spec);
        let model = params.iter().find(|p| p.key == "model").unwrap();
        let Kind::Choice { options, default } = &model.kind else { panic!("model is a menu") };
        assert!(options.len() > 8, "{}", options.len());
        assert_eq!(default.as_deref(), Some(photocraft_engine::ai_cmds::DEFAULT_MODEL));
        assert!(options.iter().any(|id| id == photocraft_engine::ai_cmds::DEFAULT_MODEL));
        assert!(params.iter().any(|p| p.key == "context" && matches!(p.kind, Kind::Json)));
        assert!(params.iter().any(|p| p.key == "aspectRatio"));
    }

    #[test]
    fn links_need_http_and_no_password() {
        assert!(acceptable_link("https://example.com/a.png"));
        assert!(acceptable_link("http://example.com/a.jpg"));
        assert!(!acceptable_link("file:///tmp/a.png"));
        assert!(!acceptable_link("https://user:secret@example.com/a.png"));
        assert!(!acceptable_link("https://"));
        assert!(!looks_like_image("notes.txt", b"hello"));
        assert!(looks_like_image("photo.jpg", b"not-yet"));
        assert!(looks_like_image("noext", b"\x89PNG\r\n"));
    }
}
