//! Edit › Generate Image and Edit › Generative Fill.
//!
//! Both commands call [OpenRouter's image API](https://openrouter.ai/docs/guides/overview/multimodal/image-generation).
//! The key is `OPENROUTER_MAIN_KEY`, or `OPENROUTER_API_KEY` when that is the only one set, in the
//! environment or in a `.env` file in the directory PhotoCraft was started from. When both are
//! set, `OPENROUTER_MAIN_KEY` wins. The key is never written into a document, a log line or an error.
//!
//! Generate Image makes a new picture from a prompt. With a document open it becomes a new layer;
//! otherwise it opens a new document. Generative Fill sends the canvas (or the selection, plus a
//! little of the picture around it) and puts the model's picture on a new layer. A selection
//! becomes that layer's mask, so the change shows only inside the selection.
//!
//! Both commands take a model id from [`IMAGE_MODELS`] and up to [`MAX_CONTEXT`] extra pictures
//! in `context`: an open document, one of its layers, a file path, image bytes, or an http(s) link.

use photocraft_color::PixelFormat;
use photocraft_doc::{Document, Layer, LayerContent, LayerMask};
use photocraft_geom::Rect;
use photocraft_raster::Surface;
use serde_json::{Value, json};

use crate::commands::CommandSpec;
use crate::{EngineError, Result, Session};

const FILL: &str = "edit.generativeFill";
const GENERATE: &str = "edit.generateImage";
/// Fast model that accepts a prompt and a reference image. A stronger id can be passed as `model`.
pub const DEFAULT_MODEL: &str = "google/gemini-3.1-flash-lite-image";
/// How many extra pictures one call may send, besides the picture Generative Fill is editing.
pub const MAX_CONTEXT: usize = 4;
/// OpenRouter image models the dialog offers. A command may still name another id.
pub const IMAGE_MODELS: &[&str] = &[
    "black-forest-labs/flux-3-image",
    "black-forest-labs/flux-3-image-klein",
    "black-forest-labs/flux.2-flex",
    "black-forest-labs/flux.2-klein-4b",
    "black-forest-labs/flux.2-max",
    "black-forest-labs/flux.2-pro",
    "bytedance-seed/seedream-4.5",
    "bytedance-seed/seedream-5-0-flash",
    "bytedance-seed/seedream-5-0-lite",
    "bytedance-seed/seedream-5-0-pro",
    "google/gemini-2.5-flash-image",
    "google/gemini-3-pro-image",
    "google/gemini-3-pro-image-preview",
    "google/gemini-3.1-flash-image",
    "google/gemini-3.1-flash-image-preview",
    "google/gemini-3.1-flash-lite-image",
    "google/gemini-nano-banana-2.1",
    "krea/krea-2-large",
    "krea/krea-2-medium",
    "krea/krea-2-medium-turbo",
    "meta/muse-image",
    "microsoft/mai-image-2.5",
    "microsoft/mai-image-2.5-pro",
    "microsoft/mai-image-2.6",
    "microsoft/mai-image-2.6-flash",
    "openai/gpt-5-image",
    "openai/gpt-5-image-mini",
    "openai/gpt-5.4-image-2",
    "openai/gpt-image-1",
    "openai/gpt-image-1-mini",
    "openai/gpt-image-2",
    "openai/gpt-image-2.5-flare",
    "openai/gpt-image-2.5-sunburst",
    "qwen/qwen-image-3",
    "qwen/qwen-image-3-pro",
    "recraft/recraft-v3",
    "recraft/recraft-v4",
    "recraft/recraft-v4-pro",
    "recraft/recraft-v4-pro-vector",
    "recraft/recraft-v4-styles",
    "recraft/recraft-v4-styles-pro",
    "recraft/recraft-v4-styles-pro-vector",
    "recraft/recraft-v4-styles-vector",
    "recraft/recraft-v4-vector",
    "recraft/recraft-v4.1",
    "recraft/recraft-v4.1-flash",
    "recraft/recraft-v4.1-pro",
    "recraft/recraft-v4.1-pro-vector",
    "recraft/recraft-v4.1-utility",
    "recraft/recraft-v4.1-utility-pro",
    "recraft/recraft-v4.1-vector",
    "sourceful/riverflow-v2-fast",
    "sourceful/riverflow-v2-pro",
    "sourceful/riverflow-v2.5-fast",
    "sourceful/riverflow-v2.5-pro",
    "tencent/hy-image-v3.5-preview",
    "x-ai/grok-imagine-image-2.0",
    "x-ai/grok-imagine-image-quality",
];
const ASPECTS: &[&str] = &["1:1", "16:9", "9:16", "4:3", "3:2", "3:4"];
/// Longest edge of the picture sent to the model. The result is scaled back to the canvas.
const MAX_EDGE: u32 = 1024;
const MAX_PROMPT: usize = 4000;
const MAX_BODY: u64 = 32 << 20;
const MAX_IMAGE: usize = 32 << 20;
const MAX_SIDE: u32 = 8192;
/// Extra canvas around a selection, so the model can see what the selection sits in.
const SELECTION_MARGIN: i32 = 48;
const API_URL: &str = "https://openrouter.ai/api/v1/images";

fn other(msg: impl Into<String>) -> EngineError {
    EngineError::Other(msg.into())
}

fn bad(cmd: &str, msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: cmd.into(), msg: msg.into() }
}

fn has_doc(s: &Session) -> std::result::Result<(), String> {
    s.active().map(|_| ()).ok_or_else(|| "no document open".into())
}

fn always(_: &Session) -> std::result::Result<(), String> {
    Ok(())
}

/// Where a finished image lands.
struct Plan {
    /// `None` opens a new document at the image's own size.
    dest: Option<Dest>,
    label: &'static str,
    /// Document title when the image opens on its own. Not the API key.
    title: String,
}

struct Dest {
    /// Stretch the image to this rectangle, or fit it inside when `fit` is set.
    rect: Rect,
    fit: bool,
    mask: Option<Surface>,
    name: String,
}

struct Request {
    prompt: String,
    model: String,
    /// Sent only for a new picture. An edit follows the reference images.
    aspect: Option<String>,
    /// Generative Fill: the first picture is the one being edited.
    edit: bool,
    sources: Vec<RefSource>,
}

/// One picture to send with the prompt. Document pixels are already encoded; the rest is read
/// while the request runs.
enum RefSource {
    Png(Vec<u8>),
    Path(String),
    Inline { name: String, bytes: Vec<u8> },
    Url(String),
}

struct Made {
    png: Vec<u8>,
    model: String,
    cost: Option<f64>,
}

/// Whether `id` is Generate Image or Generative Fill.
pub fn is_ai_command(id: &str) -> bool {
    id == FILL || id == GENERATE
}

pub fn specs() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: FILL,
            label: "Generative Fill…",
            menu: &["Edit"],
            shortcut: None,
            params: fill_params(),
            enabled: has_doc,
            journal: true,
            run: |s, p| run(s, FILL, p),
        },
        CommandSpec {
            id: GENERATE,
            label: "Generate Image…",
            menu: &["Edit"],
            shortcut: None,
            params: generate_params(),
            enabled: always,
            journal: true,
            run: |s, p| run(s, GENERATE, p),
        },
    ]
}

fn model_menu() -> String {
    IMAGE_MODELS.join("|")
}

fn fill_params() -> &'static str {
    static PARAMS: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    PARAMS
        .get_or_init(|| {
            format!(
                r#"{{"prompt":text,"model":"{models}"="{DEFAULT_MODEL}","context":json}} required prompt. model is an OpenRouter image model id. context is up to {MAX_CONTEXT} extra images, each {{"document":index}}, {{"document":index,"layer":id}}, {{"path":file}}, {{"url":"https://…"}}, or {{"name":file,"data":base64}}. Needs OPENROUTER_MAIN_KEY or OPENROUTER_API_KEY. Sends the canvas, or the selection plus surrounding context, and adds the result as a new layer."#,
                models = model_menu(),
            )
        })
        .as_str()
}

fn generate_params() -> &'static str {
    static PARAMS: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    PARAMS
        .get_or_init(|| {
            format!(
                r#"{{"prompt":text,"model":"{models}"="{DEFAULT_MODEL}","aspectRatio":"1:1|16:9|9:16|4:3|3:2|3:4"="1:1","context":json}} required prompt. model is an OpenRouter image model id. aspectRatio applies to the new picture. context is up to {MAX_CONTEXT} images, each {{"document":index}}, {{"document":index,"layer":id}}, {{"path":file}}, {{"url":"https://…"}}, or {{"name":file,"data":base64}}. Needs OPENROUTER_MAIN_KEY or OPENROUTER_API_KEY. Adds a layer, or opens a new document when none is open."#,
                models = model_menu(),
            )
        })
        .as_str()
}

fn run(s: &mut Session, cmd: &str, p: &Value) -> Result<Value> {
    let (request, plan) = prepare(s, cmd, p)?;
    let lock = plan.dest.is_some();
    let label = plan.label;
    crate::jobs::run(s, label, lock, move |ctx| fetch(ctx, &request), move |s, made| apply(s, made, plan))
}

fn prepare(s: &Session, cmd: &str, p: &Value) -> Result<(Request, Plan)> {
    let mut request = parse_request(cmd, p)?;
    request.sources.extend(context_sources(s, cmd, p)?);
    if cmd == GENERATE && s.active().is_none() {
        let title = request.prompt.clone();
        return Ok((request, Plan { dest: None, label: "Generate Image", title }));
    }
    let st = s.active().ok_or(EngineError::NoDocument)?;
    let canvas = st.doc.bounds();
    if canvas.is_empty() {
        return Err(other("the document has no pixels"));
    }
    if cmd == GENERATE {
        let title = request.prompt.clone();
        let plan = Plan { dest: Some(Dest { rect: canvas, fit: true, mask: None, name: "Generated".into() }), label: "Generate Image", title };
        return Ok((request, plan));
    }
    let selection = st.doc.selection.as_ref().map(Surface::content_bounds).filter(|r| !r.is_empty());
    let (rect, mask) = if let Some(sel) = selection {
        let rect = inflate(sel, SELECTION_MARGIN).intersect(&canvas);
        (rect, st.doc.selection.clone().map(|m| m.convert(PixelFormat::GRAY8)))
    } else {
        (canvas, None)
    };
    if rect.is_empty() {
        return Err(other("the selection is outside the canvas"));
    }
    request.edit = true;
    request.sources.insert(0, RefSource::Png(upload_png(&st.doc, rect)?));
    let title = request.prompt.clone();
    let plan = Plan { dest: Some(Dest { rect, fit: false, mask, name: "Generative Fill".into() }), label: "Generative Fill", title };
    Ok((request, plan))
}

fn parse_request(cmd: &str, p: &Value) -> Result<Request> {
    let prompt = match p.get("prompt") {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(s)) => s.trim().to_string(),
        Some(_) => return Err(bad(cmd, "`prompt` must be a string")),
    };
    if prompt.is_empty() {
        return Err(bad(cmd, "`prompt` must be a non-empty string"));
    }
    if prompt.chars().count() > MAX_PROMPT {
        return Err(bad(cmd, format!("`prompt` is longer than {MAX_PROMPT} characters")));
    }
    let model = match p.get("model") {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(s)) => s.trim().to_string(),
        Some(_) => return Err(bad(cmd, "`model` must be a string")),
    };
    let aspect = if cmd == GENERATE { Some(aspect_ratio(cmd, p)?) } else { None };
    Ok(Request { prompt, model: model_slug(cmd, &model)?, aspect, edit: false, sources: Vec::new() })
}

fn model_slug(cmd: &str, raw: &str) -> Result<String> {
    let model = if raw.is_empty() { DEFAULT_MODEL.to_string() } else { raw.to_string() };
    let ok = model.len() <= 80
        && model.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-' | '/' | ':'))
        && !model.starts_with('/')
        && !model.ends_with('/')
        && !model.contains("//");
    if !ok {
        return Err(bad(cmd, "model must be an OpenRouter model id such as google/gemini-3.1-flash-image"));
    }
    Ok(model)
}

fn aspect_ratio(cmd: &str, p: &Value) -> Result<String> {
    let ratio = match p.get("aspectRatio") {
        None | Some(Value::Null) => "1:1".to_string(),
        Some(Value::String(s)) => s.trim().to_string(),
        Some(_) => return Err(bad(cmd, "`aspectRatio` must be a string")),
    };
    if !ASPECTS.contains(&ratio.as_str()) {
        return Err(bad(cmd, format!("`aspectRatio` must be one of {}", ASPECTS.join(", "))));
    }
    Ok(ratio)
}

fn inflate(r: Rect, margin: i32) -> Rect {
    Rect::new(r.x0.saturating_sub(margin), r.y0.saturating_sub(margin), r.x1.saturating_add(margin), r.y1.saturating_add(margin))
}

fn layer_png(layer: &Layer) -> Result<Vec<u8>> {
    let surf = layer.surface().ok_or_else(|| other("that layer has no pixels"))?;
    let rect = surf.content_bounds();
    if rect.is_empty() {
        return Err(other("that layer has no pixels"));
    }
    png_of_surface(surf, rect)
}

fn upload_png(doc: &Document, rect: Rect) -> Result<Vec<u8>> {
    let surf = crate::pixels::composite_layers(doc, PixelFormat::RGBA8, Some([1.0, 1.0, 1.0]));
    png_of_surface(&surf, rect)
}

fn png_of_surface(surf: &Surface, rect: Rect) -> Result<Vec<u8>> {
    let (tw, th) = sample_size(rect.width(), rect.height())?;
    let mut rgb = Vec::new();
    let pixels = (tw as usize).checked_mul(th as usize).ok_or_else(|| other("the image is too large to send"))?;
    rgb.try_reserve(pixels.saturating_mul(3)).map_err(|_| other("not enough memory to prepare the image"))?;
    for y in 0..th {
        let sy = sample_at(rect.y0, y, rect.height(), th)?;
        for x in 0..tw {
            let sx = sample_at(rect.x0, x, rect.width(), tw)?;
            rgb.extend_from_slice(&matte(surf.rgba(sx, sy)));
        }
    }
    let img = photocraft_codecs::Image::from_u8(tw, th, photocraft_codecs::ChannelLayout::Rgb, rgb)
        .map_err(|e| other(format!("could not encode the image ({e})")))?;
    photocraft_codecs::encode(&img, photocraft_codecs::Format::Png, &photocraft_codecs::EncodeOptions::default())
        .map_err(|e| other(format!("could not encode the image ({e})")))
}

fn sample_size(w: u32, h: u32) -> Result<(u32, u32)> {
    if w == 0 || h == 0 {
        return Err(other("nothing to send to the image model"));
    }
    let long = w.max(h);
    if long <= MAX_EDGE {
        return Ok((w, h));
    }
    let scale = f64::from(MAX_EDGE) / f64::from(long);
    if !scale.is_finite() || scale <= 0.0 {
        return Err(other("could not scale the image for the model"));
    }
    let tw = (f64::from(w) * scale).round().clamp(1.0, f64::from(MAX_EDGE)) as u32;
    let th = (f64::from(h) * scale).round().clamp(1.0, f64::from(MAX_EDGE)) as u32;
    Ok((tw, th))
}

fn sample_at(origin: i32, index: u32, span: u32, samples: u32) -> Result<i32> {
    if samples == 0 {
        return Err(other("nothing to send to the image model"));
    }
    let delta = u64::from(index) * u64::from(span) / u64::from(samples);
    let delta = i32::try_from(delta).map_err(|_| other("image coordinates overflow"))?;
    origin.checked_add(delta).ok_or_else(|| other("image coordinates overflow"))
}

fn matte(p: [f32; 4]) -> [u8; 3] {
    let a = if p[3].is_finite() { p[3].clamp(0.0, 1.0) } else { 0.0 };
    let ch = |v: f32| {
        let v = if v.is_finite() { v } else { 0.0 };
        (v * a + (1.0 - a)).clamp(0.0, 1.0)
    };
    let byte = |v: f32| (v * 255.0).round() as u8;
    [byte(ch(p[0])), byte(ch(p[1])), byte(ch(p[2]))]
}

fn fetch(ctx: &crate::jobs::JobCtx, request: &Request) -> Result<Made> {
    ctx.check()?;
    ctx.progress(0.05, "Contacting the image model");
    #[cfg(target_arch = "wasm32")]
    {
        let _ = request;
        return Err(other("Generate Image and Generative Fill run in the native app, not in the browser"));
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let key = api_key()?;
        ctx.progress(0.12, "Reading context images");
        let pngs = resolve_sources(&request.sources)?;
        ctx.check()?;
        ctx.progress(0.2, "Waiting for the image model");
        let body = request_body(request, &pngs)?;
        let text = post_images(&key, &body)?;
        ctx.check()?;
        ctx.progress(0.9, "Reading the image");
        let (png, cost) = image_bytes(&text)?;
        Ok(Made { png, model: request.model.clone(), cost })
    }
}

fn request_body(request: &Request, pngs: &[Vec<u8>]) -> Result<Value> {
    let instruction = if request.edit && pngs.len() > 1 {
        format!(
            "Edit the first attached photograph. Keep the same framing, subject and lighting except for this change: {}. Any further attached images are reference only.",
            request.prompt
        )
    } else if request.edit {
        format!("Edit the attached photograph. Keep the same framing, subject and lighting except for this change: {}", request.prompt)
    } else if !pngs.is_empty() {
        format!("Use the attached images as context. {}", request.prompt)
    } else {
        request.prompt.clone()
    };
    let mut body = json!({
        "model": request.model,
        "prompt": instruction,
        "n": 1,
        "output_format": "png",
    });
    if let Some(ratio) = &request.aspect
        && let Some(obj) = body.as_object_mut()
    {
        obj.insert("aspect_ratio".into(), json!(ratio));
    }
    if !pngs.is_empty()
        && let Some(obj) = body.as_object_mut()
    {
        let refs: Vec<Value> = pngs
            .iter()
            .map(|png| {
                let mut url = String::from("data:image/png;base64,");
                url.push_str(&photocraft_paint::tile::b64_encode(png));
                json!({ "type": "image_url", "image_url": { "url": url } })
            })
            .collect();
        obj.insert("input_references".into(), Value::Array(refs));
    }
    Ok(body)
}

fn context_sources(s: &Session, cmd: &str, p: &Value) -> Result<Vec<RefSource>> {
    let Some(value) = p.get("context").filter(|v| !v.is_null()) else { return Ok(Vec::new()) };
    let items = value.as_array().ok_or_else(|| bad(cmd, "`context` must be a list of images"))?;
    if items.len() > MAX_CONTEXT {
        return Err(bad(cmd, format!("at most {MAX_CONTEXT} context images")));
    }
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        out.push(one_source(s, cmd, item)?);
    }
    Ok(out)
}

fn one_source(s: &Session, cmd: &str, item: &Value) -> Result<RefSource> {
    let obj = item.as_object().ok_or_else(|| bad(cmd, "each context image must be an object"))?;
    if let Some(url) = obj.get("url").filter(|v| !v.is_null()) {
        let url = url.as_str().ok_or_else(|| bad(cmd, "`url` must be a string"))?;
        return Ok(RefSource::Url(image_link(cmd, url)?));
    }
    if let Some(path) = obj.get("path").filter(|v| !v.is_null()) {
        let path = path.as_str().ok_or_else(|| bad(cmd, "`path` must be a string"))?;
        return Ok(RefSource::Path(image_path(cmd, path)?));
    }
    if let Some(data) = obj.get("data").filter(|v| !v.is_null()) {
        let data = data.as_str().ok_or_else(|| bad(cmd, "`data` must be base64"))?;
        let name = obj.get("name").and_then(Value::as_str).unwrap_or("context.png");
        return Ok(RefSource::Inline { name: file_name(name), bytes: image_data(cmd, data)? });
    }
    if let Some(index) = obj.get("document").filter(|v| !v.is_null()) {
        let index = index.as_u64().ok_or_else(|| bad(cmd, "`document` must be a document index"))?;
        let index = usize::try_from(index).map_err(|_| bad(cmd, "`document` is out of range"))?;
        let st = s.documents().get(index).ok_or_else(|| bad(cmd, format!("no document at index {index}")))?;
        let png = if let Some(layer) = obj.get("layer").filter(|v| !v.is_null()) {
            let id = layer.as_u64().ok_or_else(|| bad(cmd, "`layer` must be a layer id"))?;
            let layer = st.doc.layer(photocraft_doc::LayerId(id)).ok_or_else(|| bad(cmd, format!("no layer {id}")))?;
            layer_png(layer)?
        } else {
            let canvas = st.doc.bounds();
            if canvas.is_empty() {
                return Err(bad(cmd, "that document has no pixels"));
            }
            upload_png(&st.doc, canvas)?
        };
        return Ok(RefSource::Png(png));
    }
    Err(bad(cmd, "a context image needs a document, path, url, or data"))
}

fn image_link(cmd: &str, raw: &str) -> Result<String> {
    let url = raw.trim();
    let rest = url.strip_prefix("https://").or_else(|| url.strip_prefix("http://"));
    let Some(rest) = rest else {
        return Err(bad(cmd, "a context image link must start with http:// or https://"));
    };
    if url.len() > 2000 || rest.is_empty() || url.chars().any(|c| c.is_control() || c.is_whitespace()) {
        return Err(bad(cmd, "that context image link is not usable"));
    }
    let host = rest.split(['/', '?', '#']).next().unwrap_or("");
    if host.is_empty() || host.contains('@') {
        return Err(bad(cmd, "that context image link is not usable"));
    }
    Ok(url.to_string())
}

fn image_path(cmd: &str, raw: &str) -> Result<String> {
    let path = raw.trim();
    if path.is_empty() || path.len() > 4096 || path.chars().any(|c| c == '\0') {
        return Err(bad(cmd, "that context image path is not usable"));
    }
    Ok(path.to_string())
}

fn image_data(cmd: &str, data: &str) -> Result<Vec<u8>> {
    if data.len() > MAX_IMAGE.saturating_mul(2) {
        return Err(bad(cmd, "a context image is larger than 32 MB"));
    }
    let bytes = photocraft_paint::tile::b64_decode(data).ok_or_else(|| bad(cmd, "a context image was not valid base64"))?;
    if bytes.is_empty() || bytes.len() > MAX_IMAGE {
        return Err(bad(cmd, "a context image was empty or larger than 32 MB"));
    }
    Ok(bytes)
}

fn file_name(name: &str) -> String {
    let name = name.rsplit(['/', '\\']).next().unwrap_or(name).trim();
    let mut out = String::new();
    for ch in name.chars().take(80) {
        if ch.is_control() {
            continue;
        }
        out.push(ch);
    }
    if out.is_empty() { "context.png".into() } else { out }
}

fn resolve_sources(sources: &[RefSource]) -> Result<Vec<Vec<u8>>> {
    let mut out = Vec::with_capacity(sources.len());
    for source in sources {
        out.push(resolve_source(source)?);
    }
    Ok(out)
}

fn resolve_source(source: &RefSource) -> Result<Vec<u8>> {
    match source {
        RefSource::Png(bytes) => Ok(bytes.clone()),
        RefSource::Inline { name, bytes } => png_of_bytes(name, bytes),
        RefSource::Path(path) => {
            #[cfg(target_arch = "wasm32")]
            {
                let _ = path;
                Err(other("a context image from a file runs in the native app, not in the browser"))
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                let bytes = read_image_file(std::path::Path::new(path))?;
                png_of_bytes(&file_name(path), &bytes)
            }
        }
        RefSource::Url(url) => {
            #[cfg(target_arch = "wasm32")]
            {
                let _ = url;
                Err(other("a context image from a link runs in the native app, not in the browser"))
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                let bytes = download_image(url)?;
                png_of_bytes("context.png", &bytes)
            }
        }
    }
}

fn png_of_bytes(name: &str, bytes: &[u8]) -> Result<Vec<u8>> {
    let imported = photocraft_io::import(name, bytes).map_err(|_| other(format!("{name} is not an image PhotoCraft can read")))?;
    check_side(imported.document.size.width, imported.document.size.height)?;
    let canvas = imported.document.bounds();
    if canvas.is_empty() {
        return Err(other(format!("{name} has no pixels")));
    }
    upload_png(&imported.document, canvas)
}

#[cfg(not(target_arch = "wasm32"))]
fn read_image_file(path: &std::path::Path) -> Result<Vec<u8>> {
    use std::io::Read;
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("image");
    let file = std::fs::File::open(path).map_err(|_| other(format!("could not read {name}")))?;
    let mut file = file.take(MAX_IMAGE as u64 + 1);
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).map_err(|_| other(format!("could not read {name}")))?;
    if buf.is_empty() || buf.len() > MAX_IMAGE {
        return Err(other(format!("{name} was empty or larger than 32 MB")));
    }
    Ok(buf)
}

#[cfg(not(target_arch = "wasm32"))]
fn download_image(url: &str) -> Result<Vec<u8>> {
    use std::io::Read;
    let agent = ureq::AgentBuilder::new().timeout(std::time::Duration::from_secs(30)).redirects(3).build();
    let response = agent.get(url).set("Accept", "image/*,*/*").call();
    let resp = match response {
        Ok(resp) => resp,
        Err(ureq::Error::Status(code, _)) => return Err(other(format!("could not download the context image (HTTP {code})"))),
        Err(_) => return Err(other("could not download the context image")),
    };
    let mut buf = Vec::new();
    resp.into_reader().take(MAX_IMAGE as u64 + 1).read_to_end(&mut buf).map_err(|_| other("could not download the context image"))?;
    if buf.is_empty() || buf.len() > MAX_IMAGE {
        return Err(other("the downloaded context image was empty or larger than 32 MB"));
    }
    Ok(buf)
}

fn apply(s: &mut Session, made: Made, plan: Plan) -> Result<Value> {
    let mut out = json!({"model": made.model});
    if let Some(cost) = made.cost
        && let Some(obj) = out.as_object_mut()
    {
        obj.insert("costUsd".into(), json!(cost));
    }
    match plan.dest {
        None => {
            let mut imported = photocraft_io::import("Generated.png", &made.png)
                .map_err(|e| other(format!("the image model returned data PhotoCraft could not read ({e})")))?;
            check_side(imported.document.size.width, imported.document.size.height)?;
            imported.document.name = short_name(&plan.title, 48);
            if let Some(layer) = imported.document.layers.first_mut() {
                layer.name = "Generated".into();
            }
            let w = imported.document.size.width;
            let h = imported.document.size.height;
            let index = s.add_document(imported.document, None);
            if let Some(obj) = out.as_object_mut() {
                obj.insert("document".into(), json!(index));
                obj.insert("width".into(), json!(w));
                obj.insert("height".into(), json!(h));
            }
        }
        Some(dest) => {
            let (surf, w, h) = raster_from_png(&made.png)?;
            let placed = place(&surf, w, h, &dest)?;
            let id = s.edit(plan.label, |doc, active| {
                let placed = placed.try_convert(doc.pixel_format())?;
                let name = layer_name(doc, &dest.name);
                let mut layer = Layer::new(name, LayerContent::Raster(placed));
                if let Some(mask) = dest.mask.clone() {
                    layer.mask = Some(LayerMask { surface: mask, enabled: true, linked: true, density: 1.0, feather: 0.0 });
                }
                let id = doc.insert_above(*active, layer);
                *active = Some(id);
                Ok(id)
            })?;
            if let Some(obj) = out.as_object_mut() {
                obj.insert("layer".into(), json!(id.0));
                obj.insert("width".into(), json!(w));
                obj.insert("height".into(), json!(h));
            }
        }
    }
    Ok(out)
}

fn raster_from_png(bytes: &[u8]) -> Result<(Surface, u32, u32)> {
    let imported =
        photocraft_io::import("generated.png", bytes).map_err(|e| other(format!("the image model returned data PhotoCraft could not read ({e})")))?;
    let w = imported.document.size.width;
    let h = imported.document.size.height;
    check_side(w, h)?;
    let layer = imported.document.layers.into_iter().next().ok_or_else(|| other("the image model returned an empty image"))?;
    let LayerContent::Raster(surf) = layer.content else {
        return Err(other("the image model returned an image with no pixels"));
    };
    Ok((surf, w, h))
}

fn check_side(w: u32, h: u32) -> Result<()> {
    if w == 0 || h == 0 || w > MAX_SIDE || h > MAX_SIDE {
        return Err(other(format!("the generated image is {w}×{h}; PhotoCraft accepts up to {MAX_SIDE} px on a side")));
    }
    Ok(())
}

fn place(surf: &Surface, img_w: u32, img_h: u32, dest: &Dest) -> Result<Surface> {
    let target = if dest.fit { fit_inside(img_w, img_h, dest.rect)? } else { dest.rect };
    let dw = target.width();
    let dh = target.height();
    if img_w == 0 || img_h == 0 || dw == 0 || dh == 0 {
        return Err(other("the generated image has no pixels"));
    }
    let sx = f64::from(dw) / f64::from(img_w);
    let sy = f64::from(dh) / f64::from(img_h);
    if !sx.is_finite() || !sy.is_finite() || sx <= 0.0 || sy <= 0.0 {
        return Err(other("could not scale the generated image"));
    }
    let scaled = photocraft_algo::resample::resize_surface(surf, sx, sy, photocraft_algo::resample::Resample::Bilinear);
    let bounds = scaled.content_bounds();
    let dw_i = i32::try_from(dw).map_err(|_| other("the canvas is too large"))?;
    let dh_i = i32::try_from(dh).map_err(|_| other("the canvas is too large"))?;
    let content = if bounds.is_empty() { Rect::new(0, 0, dw_i, dh_i) } else { bounds };
    Ok(scaled.translated(target.x0, target.y0, content))
}

fn fit_inside(img_w: u32, img_h: u32, canvas: Rect) -> Result<Rect> {
    let cw = canvas.width();
    let ch = canvas.height();
    if img_w == 0 || img_h == 0 || cw == 0 || ch == 0 {
        return Err(other("the canvas has no pixels"));
    }
    let scale = (f64::from(cw) / f64::from(img_w)).min(f64::from(ch) / f64::from(img_h));
    if !scale.is_finite() || scale <= 0.0 {
        return Err(other("could not fit the generated image"));
    }
    let dw = (f64::from(img_w) * scale).round().clamp(1.0, f64::from(cw)) as i32;
    let dh = (f64::from(img_h) * scale).round().clamp(1.0, f64::from(ch)) as i32;
    let x = canvas.x0 + (i32::try_from(cw).unwrap_or(0) - dw) / 2;
    let y = canvas.y0 + (i32::try_from(ch).unwrap_or(0) - dh) / 2;
    Ok(Rect::new(x, y, x.saturating_add(dw), y.saturating_add(dh)))
}

fn layer_name(doc: &Document, base: &str) -> String {
    let taken = doc.walk().into_iter().any(|(_, _, layer)| layer.name == base);
    if taken { doc.next_layer_name(base) } else { base.to_string() }
}

fn short_name(text: &str, max: usize) -> String {
    let mut out = String::new();
    for (i, ch) in text.chars().enumerate() {
        if i >= max {
            break;
        }
        out.push(ch);
    }
    if out.is_empty() { "Generated".into() } else { out }
}

/// Inference key first, then the older name. A `.env` often keeps a management key in
/// `OPENROUTER_API_KEY` and the key that can generate images in `OPENROUTER_MAIN_KEY`.
const KEY_NAMES: &[&str] = &["OPENROUTER_MAIN_KEY", "OPENROUTER_API_KEY"];

/// `OPENROUTER_MAIN_KEY` or `OPENROUTER_API_KEY` from the environment, then from `./.env`.
fn api_key() -> Result<String> {
    if let Some(key) = key_from_env() {
        return Ok(key);
    }
    if let Ok(dir) = std::env::current_dir()
        && let Ok(text) = read_text_limited(&dir.join(".env"))
        && let Some(key) = key_from_dotenv(&text)
    {
        return Ok(key);
    }
    Err(other("Set OPENROUTER_MAIN_KEY or OPENROUTER_API_KEY, or start PhotoCraft from a directory whose .env file defines one of them."))
}

fn key_from_env() -> Option<String> {
    KEY_NAMES.iter().find_map(|name| std::env::var(name).ok().and_then(|value| nonempty(&value)))
}

fn key_from_dotenv(text: &str) -> Option<String> {
    KEY_NAMES.iter().find_map(|name| dotenv_value(text, name))
}

fn nonempty(value: &str) -> Option<String> {
    let trimmed = value.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

fn read_text_limited(path: &std::path::Path) -> std::io::Result<String> {
    use std::io::Read;
    let file = std::fs::File::open(path)?;
    let mut buf = Vec::new();
    file.take(64 * 1024).read_to_end(&mut buf)?;
    String::from_utf8(buf).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

/// Value of `name` in a dotenv file. Later lines win. Quotes around the value are removed.
fn dotenv_value(text: &str, name: &str) -> Option<String> {
    let mut found = None;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let line = line.strip_prefix("export ").unwrap_or(line);
        let Some((key, value)) = line.split_once('=') else { continue };
        if key.trim() != name {
            continue;
        }
        let value = value.trim();
        let value =
            value.strip_prefix('"').and_then(|v| v.strip_suffix('"')).or_else(|| value.strip_prefix('\'').and_then(|v| v.strip_suffix('\''))).unwrap_or(value);
        if let Some(key) = nonempty(value) {
            found = Some(key);
        }
    }
    found
}

fn image_bytes(body: &str) -> Result<(Vec<u8>, Option<f64>)> {
    let v: Value = serde_json::from_str(body).map_err(|_| other("OpenRouter returned a response that was not JSON"))?;
    if let Some(msg) = v.get("error").and_then(|e| e.get("message")).and_then(Value::as_str) {
        return Err(other(format!("OpenRouter: {}", clip(msg))));
    }
    let item = v.get("data").and_then(Value::as_array).and_then(|a| a.first()).ok_or_else(|| other("OpenRouter returned no image"))?;
    let media = item.get("media_type").and_then(Value::as_str).unwrap_or("");
    if media.contains("svg") {
        return Err(other("the model returned SVG; choose a raster image model"));
    }
    let b64 = item.get("b64_json").and_then(Value::as_str).ok_or_else(|| other("OpenRouter returned no image"))?;
    let bytes = photocraft_paint::tile::b64_decode(b64).ok_or_else(|| other("OpenRouter returned an image that was not valid base64"))?;
    if bytes.is_empty() || bytes.len() > MAX_IMAGE {
        return Err(other("the generated image was empty or larger than 32 MB"));
    }
    let cost = v.get("usage").and_then(|u| u.get("cost")).and_then(Value::as_f64).filter(|c| c.is_finite());
    Ok((bytes, cost))
}

fn clip(text: &str) -> String {
    let flat: String = text.chars().map(|c| if c.is_control() { ' ' } else { c }).take(300).collect();
    flat.trim().to_string()
}

#[cfg(not(target_arch = "wasm32"))]
fn post_images(key: &str, body: &Value) -> Result<String> {
    let payload = serde_json::to_string(body).map_err(|_| other("could not build the image request"))?;
    let agent = ureq::AgentBuilder::new().timeout(std::time::Duration::from_secs(180)).build();
    let mut auth = String::from("Bearer ");
    auth.push_str(key);
    let response = agent
        .post(API_URL)
        .set("Authorization", &auth)
        .set("Content-Type", "application/json")
        .set("Accept", "application/json")
        .set("HTTP-Referer", "https://github.com/storytold/photocraft")
        .set("X-Title", "PhotoCraft")
        .send_string(&payload);
    match response {
        Ok(resp) => read_body(resp),
        Err(ureq::Error::Status(code, resp)) => {
            let text = read_body(resp).unwrap_or_else(|_| String::new());
            Err(http_error(code, &text))
        }
        Err(e) => Err(other(format!("could not reach OpenRouter: {e}"))),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn read_body(resp: ureq::Response) -> Result<String> {
    use std::io::Read;
    let mut buf = Vec::new();
    resp.into_reader().take(MAX_BODY).read_to_end(&mut buf).map_err(|e| other(format!("reading the image response failed: {e}")))?;
    if buf.len() as u64 >= MAX_BODY {
        return Err(other("the image response was larger than 32 MB"));
    }
    String::from_utf8(buf).map_err(|_| other("the image response was not text"))
}

#[cfg(not(target_arch = "wasm32"))]
fn http_error(code: u16, body: &str) -> EngineError {
    let detail = serde_json::from_str::<Value>(body).ok().and_then(|v| v.get("error")?.get("message")?.as_str().map(clip)).filter(|s| !s.is_empty());
    let detail = detail.unwrap_or_else(|| clip(body));
    let why = match code {
        // A management or provisioning key authenticates metadata calls and then fails
        // generation with this exact message. A normal key says the key was rejected.
        401 | 403 if detail.to_ascii_lowercase().contains("user not found") => {
            "OpenRouter rejected this key. Image generation needs a standard API key from https://openrouter.ai/settings/keys, not a management or provisioning key"
        }
        401 | 403 => "OpenRouter rejected the API key",
        402 => "OpenRouter reported that the account has no credits",
        429 => "OpenRouter is rate limiting requests",
        _ => "OpenRouter could not make the image",
    };
    if detail.is_empty() { other(format!("{why} (HTTP {code})")) } else { other(format!("{why} (HTTP {code}): {detail}")) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn a_management_key_gets_a_specific_error() {
        let err = http_error(401, r#"{"error":{"message":"User not found."}}"#);
        let text = err.to_string();
        assert!(text.contains("standard API key"), "{text}");
        assert!(text.contains("management"), "{text}");
    }

    #[test]
    fn empty_and_mistyped_prompts_fail() {
        let mut s = Session::new();
        let err = s.execute(GENERATE, json!({})).unwrap_err();
        assert!(err.to_string().contains("prompt"), "{err}");
        let err = s.execute(GENERATE, json!({"prompt": "   "})).unwrap_err();
        assert!(err.to_string().contains("prompt"), "{err}");
        let err = s.execute(GENERATE, json!({"prompt": 3})).unwrap_err();
        assert!(err.to_string().contains("string"), "{err}");
        let err = s.execute(GENERATE, json!({"prompt": "a red circle", "model": "not a model"})).unwrap_err();
        assert!(err.to_string().contains("model"), "{err}");
        let err = s.execute(GENERATE, json!({"prompt": "a red circle", "aspectRatio": "2:1"})).unwrap_err();
        assert!(err.to_string().contains("aspectRatio"), "{err}");
        let err = s.execute(FILL, json!({"prompt": "make the sky blue"})).unwrap_err();
        assert!(err.to_string().contains("document"), "{err}");
    }

    #[test]
    fn context_images_are_checked_before_any_request() {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 8, "height": 8, "background": "white"})).unwrap();
        let err = s.execute(GENERATE, json!({"prompt": "a cup", "context": {"url": "https://example.com/a.png"}})).unwrap_err();
        assert!(err.to_string().contains("list"), "{err}");
        let err = s.execute(GENERATE, json!({"prompt": "a cup", "context": [{"url": "file:///tmp/a.png"}]})).unwrap_err();
        assert!(err.to_string().contains("http"), "{err}");
        let err = s.execute(GENERATE, json!({"prompt": "a cup", "context": [{"url": "https://user:secret@example.com/a.png"}]})).unwrap_err();
        assert!(err.to_string().contains("not usable"), "{err}");
        assert!(!err.to_string().contains("secret"), "{err}");
        let many = vec![json!({"url": "https://example.com/a.png"}); 5];
        let err = s.execute(GENERATE, json!({"prompt": "a cup", "context": many})).unwrap_err();
        assert!(err.to_string().contains("4"), "{err}");
        let err = s.execute(GENERATE, json!({"prompt": "a cup", "context": [{"document": 3}]})).unwrap_err();
        assert!(err.to_string().contains("document"), "{err}");
        let err = s.execute(GENERATE, json!({"prompt": "a cup", "context": [{"document": 0, "layer": 999}]})).unwrap_err();
        assert!(err.to_string().contains("layer"), "{err}");
    }

    #[test]
    fn an_open_document_and_inline_bytes_become_reference_images() {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 4, "height": 4, "background": "white"})).unwrap();
        let png = red_png();
        let data = photocraft_paint::tile::b64_encode(&png);
        let sources = context_sources(
            &s,
            GENERATE,
            &json!({"context": [{"document": 0}, {"name": "red.png", "data": data}]}),
        )
        .unwrap();
        assert_eq!(sources.len(), 2);
        let pngs = resolve_sources(&sources).unwrap();
        assert_eq!(pngs.len(), 2);
        let request = Request { prompt: "use these".into(), model: DEFAULT_MODEL.into(), aspect: Some("1:1".into()), edit: false, sources: Vec::new() };
        let body = request_body(&request, &pngs).unwrap();
        let refs = body["input_references"].as_array().unwrap();
        assert_eq!(refs.len(), 2);
        assert!(body["prompt"].as_str().unwrap().contains("context"), "{}", body["prompt"]);
        let edit = Request { prompt: "make it blue".into(), model: DEFAULT_MODEL.into(), aspect: None, edit: true, sources: Vec::new() };
        let body = request_body(&edit, &pngs).unwrap();
        assert!(body["prompt"].as_str().unwrap().contains("first attached"), "{}", body["prompt"]);
    }

    #[test]
    fn the_model_menu_includes_the_default() {
        assert!(IMAGE_MODELS.contains(&DEFAULT_MODEL));
        assert!(IMAGE_MODELS.len() > 8);
        assert!(generate_params().contains(DEFAULT_MODEL));
        assert!(fill_params().contains("context"));
    }

    #[test]
    fn dotenv_reads_the_key_and_ignores_comments() {
        let text = "# comment\nexport OTHER=nope\nOPENROUTER_API_KEY=\"sk-test\"\n";
        assert_eq!(dotenv_value(text, "OPENROUTER_API_KEY").as_deref(), Some("sk-test"));
        assert_eq!(dotenv_value("OPENROUTER_API_KEY=\n", "OPENROUTER_API_KEY"), None);
        assert_eq!(dotenv_value("NOPE=1\n", "OPENROUTER_API_KEY"), None);
        let both = "OPENROUTER_API_KEY=sk-management\nOPENROUTER_MAIN_KEY=sk-main\n";
        assert_eq!(key_from_dotenv(both).as_deref(), Some("sk-main"));
        assert_eq!(key_from_dotenv("OPENROUTER_API_KEY=sk-only\n").as_deref(), Some("sk-only"));
    }

    #[test]
    fn response_parser_decodes_the_image_and_reports_api_errors() {
        let png = b"not-really";
        let body = json!({
            "data": [{ "b64_json": photocraft_paint::tile::b64_encode(png), "media_type": "image/png" }],
            "usage": { "cost": 0.02 }
        });
        let (bytes, cost) = image_bytes(&body.to_string()).unwrap();
        assert_eq!(bytes, png);
        assert_eq!(cost, Some(0.02));
        let err = image_bytes(r#"{"error":{"message":"no credits"}}"#).unwrap_err();
        assert!(err.to_string().contains("no credits"), "{err}");
        let err = image_bytes(r#"{"data":[{"b64_json":"!!!!"}]}"#).unwrap_err();
        assert!(err.to_string().contains("base64"), "{err}");
    }

    fn red_png() -> Vec<u8> {
        let bytes: Vec<u8> = [255u8, 0, 0, 255].into_iter().cycle().take(16).collect();
        let img = photocraft_codecs::Image::from_u8(2, 2, photocraft_codecs::ChannelLayout::Rgba, bytes).unwrap();
        photocraft_codecs::encode(&img, photocraft_codecs::Format::Png, &photocraft_codecs::EncodeOptions::default()).unwrap()
    }

    #[test]
    fn a_decoded_png_becomes_a_layer() {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 16, "height": 16})).unwrap();
        let made = Made { png: red_png(), model: DEFAULT_MODEL.into(), cost: None };
        let plan = Plan {
            dest: Some(Dest { rect: Rect::new(0, 0, 16, 16), fit: true, mask: None, name: "Generated".into() }),
            label: "Generate Image",
            title: "red".into(),
        };
        apply(&mut s, made, plan).unwrap();
        let st = s.active().unwrap();
        let layer = st.doc.layer(st.active_layer.unwrap()).unwrap();
        assert_eq!(layer.name, "Generated");
        let px = layer.surface().unwrap().rgba(8, 8);
        assert!(px[0] > 0.9 && px[1] < 0.1 && px[2] < 0.1, "{px:?}");
    }

    #[test]
    fn a_decoded_png_opens_a_document_when_none_is_open() {
        let mut s = Session::new();
        let made = Made { png: red_png(), model: DEFAULT_MODEL.into(), cost: Some(0.01) };
        let plan = Plan { dest: None, label: "Generate Image", title: "a red square".into() };
        let out = apply(&mut s, made, plan).unwrap();
        assert_eq!(out["width"], 2);
        assert_eq!(out["height"], 2);
        assert_eq!(out["costUsd"], 0.01);
        let st = s.active().unwrap();
        assert_eq!(st.doc.name, "a red square");
        assert_eq!(st.doc.size.width, 2);
        assert_eq!(st.doc.layers.first().map(|l| l.name.as_str()), Some("Generated"));
    }
}
