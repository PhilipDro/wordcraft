//! The assistant's HTTP hook: one request (POST, or GET for model lists) on its own thread, so the UI never blocks.
//! Never logs headers, URLs or bodies (they carry API keys and the user's text); only host and status.

use std::sync::mpsc;
use std::time::Duration;
use wordcraft_assistant::wire::error_message;
use wordcraft_assistant::{Http, HttpRequest, Method};

/// Longest response body read.
const MAX_BODY: u64 = 8 * 1024 * 1024;

pub fn hook(ctx: &egui::Context) -> Box<Http> {
    let ctx = ctx.clone();
    Box::new(move |req: HttpRequest| {
        let (tx, rx) = mpsc::channel();
        let ctx = ctx.clone();
        let spawned = std::thread::Builder::new().name("assistant-http".into()).spawn({
            let tx = tx.clone();
            move || {
                let _ = tx.send(post(&req));
                ctx.request_repaint();
            }
        });
        if let Err(e) = spawned {
            let _ = tx.send(Err(format!("could not start the request: {e}")));
        }
        rx
    })
}

fn post(req: &HttpRequest) -> Result<String, String> {
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .http_status_as_error(false)
        .timeout_connect(Some(Duration::from_secs(15)))
        .timeout_global(Some(Duration::from_secs(180)))
        .build()
        .into();
    let sent = match req.method {
        Method::Get => {
            let mut r = agent.get(&req.url);
            for (k, v) in &req.headers {
                r = r.header(k, v);
            }
            r.call()
        }
        Method::Post => {
            let mut r = agent.post(&req.url);
            for (k, v) in &req.headers {
                r = r.header(k, v);
            }
            r.send(req.body.as_str())
        }
    };
    let mut resp = sent.map_err(|e| format!("network error: {}", describe(&e)))?;
    let code = resp.status().as_u16();
    let body =
        resp.body_mut().with_config().limit(MAX_BODY).read_to_string().map_err(|e| format!("could not read the response: {}", describe(&e)))?;
    log::debug!("assistant request: HTTP {code}");
    if (200..300).contains(&code) { Ok(body) } else { Err(format!("HTTP {code}: {}", error_message(&body))) }
}

/// The error without its URL (ureq's messages can include it).
fn describe(e: &ureq::Error) -> String {
    match e {
        ureq::Error::Timeout(_) => "the request timed out".into(),
        ureq::Error::HostNotFound => "host not found".into(),
        ureq::Error::ConnectionFailed => "connection failed".into(),
        ureq::Error::BodyExceedsLimit(_) => "the response is too large".into(),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    /// Answers one request with `status`/`body`; returns the URL and the received raw request.
    fn serve(status: &str, body: &'static str) -> (String, mpsc::Receiver<String>) {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/v1/x", l.local_addr().unwrap());
        let (tx, rx) = mpsc::channel();
        let status = status.to_string();
        std::thread::spawn(move || {
            let (mut s, _) = l.accept().unwrap();
            let mut buf = Vec::new();
            let mut chunk = [0u8; 1024];
            loop {
                let n = s.read(&mut chunk).unwrap();
                buf.extend_from_slice(&chunk[..n]);
                let text = String::from_utf8_lossy(&buf).to_string();
                if let Some(i) = text.find("\r\n\r\n") {
                    let len = text[..i]
                        .lines()
                        .find_map(|l| l.to_ascii_lowercase().strip_prefix("content-length:").and_then(|v| v.trim().parse::<usize>().ok()))
                        .unwrap_or(0);
                    if buf.len() >= i + 4 + len {
                        break;
                    }
                }
                if n == 0 {
                    break;
                }
            }
            let _ = tx.send(String::from_utf8_lossy(&buf).to_string());
            let _ =
                write!(s, "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
        });
        (url, rx)
    }

    fn call(url: String) -> Result<String, String> {
        call_with(Method::Post, url)
    }

    fn call_with(method: Method, url: String) -> Result<String, String> {
        let h = hook(&egui::Context::default());
        let body = if method == Method::Get { String::new() } else { "{\"a\":1}".into() };
        h(HttpRequest { method, url, headers: vec![("x-api-key".into(), "k123".into())], body }).recv_timeout(Duration::from_secs(20)).unwrap()
    }

    #[test]
    fn ok_response() {
        let (url, got) = serve("200 OK", "{\"ok\":true}");
        assert_eq!(call(url).unwrap(), "{\"ok\":true}");
        let raw = got.recv().unwrap();
        assert!(raw.starts_with("POST /v1/x HTTP/1.1"), "{raw}");
        assert!(raw.to_ascii_lowercase().contains("x-api-key: k123"));
        assert!(raw.ends_with("{\"a\":1}"));
    }

    #[test]
    fn get_sends_no_body() {
        let (url, got) = serve("200 OK", "{\"data\":[]}");
        assert_eq!(call_with(Method::Get, url).unwrap(), "{\"data\":[]}");
        let raw = got.recv().unwrap();
        assert!(raw.starts_with("GET /v1/x HTTP/1.1"), "{raw}");
        assert!(raw.to_ascii_lowercase().contains("x-api-key: k123"));
        assert!(raw.ends_with("\r\n\r\n"), "{raw}");
    }

    #[test]
    fn error_status_is_readable() {
        let (url, _) = serve("401 Unauthorized", "{\"error\":{\"message\":\"invalid key\"}}");
        let e = call(url).unwrap_err();
        assert!(e.contains("401") && e.contains("invalid key"), "{e}");
    }

    #[test]
    fn nothing_listening() {
        let port = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
        let e = call(format!("http://127.0.0.1:{port}/")).unwrap_err();
        assert!(!e.is_empty());
    }
}
